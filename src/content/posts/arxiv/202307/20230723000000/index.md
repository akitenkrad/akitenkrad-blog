---
draft: false
title: "arXiv @ 2023.07.23"
date: 2023-07-23
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.23"
    identifier: arxiv_20230723
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (18)](#cslg-18)
- [cs.DL (1)](#csdl-1)
- [cs.SE (3)](#csse-3)
- [cs.NI (1)](#csni-1)
- [cs.RO (5)](#csro-5)
- [cs.SI (4)](#cssi-4)
- [cs.CV (23)](#cscv-23)
- [cs.CR (1)](#cscr-1)
- [cs.CL (9)](#cscl-9)
- [cs.AI (9)](#csai-9)
- [cs.CE (1)](#csce-1)
- [cs.DC (1)](#csdc-1)
- [cs.HC (2)](#cshc-2)
- [cs.SD (1)](#cssd-1)
- [econ.GN (1)](#econgn-1)
- [cs.MA (1)](#csma-1)
- [eess.AS (1)](#eessas-1)
- [cs.IT (1)](#csit-1)
- [physics.optics (1)](#physicsoptics-1)

## cs.LG (18)



### (1/84) Selective Perception: Optimizing State Descriptions with Reinforcement Learning for Language Model Actors (Kolby Nottingham et al., 2023)

{{<citation>}}

Kolby Nottingham, Yasaman Razeghi, Kyungmin Kim, JB Lanier, Pierre Baldi, Roy Fox, Sameer Singh. (2023)  
**Selective Perception: Optimizing State Descriptions with Reinforcement Learning for Language Model Actors**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11922v1)  

---


**ABSTRACT**  
Large language models (LLMs) are being applied as actors for sequential decision making tasks in domains such as robotics and games, utilizing their general world knowledge and planning abilities. However, previous work does little to explore what environment state information is provided to LLM actors via language. Exhaustively describing high-dimensional states can impair performance and raise inference costs for LLM actors. Previous LLM actors avoid the issue by relying on hand-engineered, task-specific protocols to determine which features to communicate about a state and which to leave out. In this work, we propose Brief Language INputs for DEcision-making Responses (BLINDER), a method for automatically selecting concise state descriptions by learning a value function for task-conditioned state descriptions. We evaluate BLINDER on the challenging video game NetHack and a robotic manipulation task. Our method improves task success rate, reduces input size and compute costs, and generalizes between LLM actors.

{{</citation>}}


### (2/84) Hindsight-DICE: Stable Credit Assignment for Deep Reinforcement Learning (Akash Velu et al., 2023)

{{<citation>}}

Akash Velu, Skanda Vaidyanath, Dilip Arumugam. (2023)  
**Hindsight-DICE: Stable Credit Assignment for Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11897v1)  

---


**ABSTRACT**  
Oftentimes, environments for sequential decision-making problems can be quite sparse in the provision of evaluative feedback to guide reinforcement-learning agents. In the extreme case, long trajectories of behavior are merely punctuated with a single terminal feedback signal, engendering a significant temporal delay between the observation of non-trivial reward and the individual steps of behavior culpable for eliciting such feedback. Coping with such a credit assignment challenge is one of the hallmark characteristics of reinforcement learning and, in this work, we capitalize on existing importance-sampling ratio estimation techniques for off-policy evaluation to drastically improve the handling of credit assignment with policy-gradient methods. While the use of so-called hindsight policies offers a principled mechanism for reweighting on-policy data by saliency to the observed trajectory return, naively applying importance sampling results in unstable or excessively lagged learning. In contrast, our hindsight distribution correction facilitates stable, efficient learning across a broad range of environments where credit assignment plagues baseline methods.

{{</citation>}}


### (3/84) JoinGym: An Efficient Query Optimization Environment for Reinforcement Learning (Kaiwen Wang et al., 2023)

{{<citation>}}

Kaiwen Wang, Junxiong Wang, Yueying Li, Nathan Kallus, Immanuel Trummer, Wen Sun. (2023)  
**JoinGym: An Efficient Query Optimization Environment for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11704v1)  

---


**ABSTRACT**  
In this paper, we present \textsc{JoinGym}, an efficient and lightweight query optimization environment for reinforcement learning (RL). Join order selection (JOS) is a classic NP-hard combinatorial optimization problem from database query optimization and can serve as a practical testbed for the generalization capabilities of RL algorithms. We describe how to formulate each of the left-deep and bushy variants of the JOS problem as a Markov Decision Process (MDP), and we provide an implementation adhering to the standard Gymnasium API. We highlight that our implementation \textsc{JoinGym} is completely based on offline traces of all possible joins, which enables RL practitioners to easily and quickly test their methods on a realistic data management problem without needing to setup any systems. Moreover, we also provide all possible join traces on $3300$ novel SQL queries generated from the IMDB dataset. Upon benchmarking popular RL algorithms, we find that at least one method can obtain near-optimal performance on train-set queries but their performance degrades by several orders of magnitude on test-set queries. This gap motivates further research for RL algorithms that generalize well in multi-task combinatorial optimization problems.

{{</citation>}}


### (4/84) Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization (Xiangsen Wang et al., 2023)

{{<citation>}}

Xiangsen Wang, Haoran Xu, Yinan Zheng, Xianyuan Zhan. (2023)  
**Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11620v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) has received considerable attention in recent years due to its attractive capability of learning policies from offline datasets without environmental interactions. Despite some success in the single-agent setting, offline multi-agent RL (MARL) remains to be a challenge. The large joint state-action space and the coupled multi-agent behaviors pose extra complexities for offline policy optimization. Most existing offline MARL studies simply apply offline data-related regularizations on individual agents, without fully considering the multi-agent system at the global level. In this work, we present OMIGA, a new offline m ulti-agent RL algorithm with implicit global-to-local v alue regularization. OMIGA provides a principled framework to convert global-level value regularization into equivalent implicit local value regularizations and simultaneously enables in-sample learning, thus elegantly bridging multi-agent value decomposition and policy learning with offline regularizations. Based on comprehensive experiments on the offline multi-agent MuJoCo and StarCraft II micro-management tasks, we show that OMIGA achieves superior performance over the state-of-the-art offline MARL methods in almost all tasks.

{{</citation>}}


### (5/84) Training Latency Minimization for Model-Splitting Allowed Federated Edge Learning (Yao Wen et al., 2023)

{{<citation>}}

Yao Wen, Guopeng Zhang, Kezhi Wang, Kun Yang. (2023)  
**Training Latency Minimization for Model-Splitting Allowed Federated Edge Learning**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-DC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11532v1)  

---


**ABSTRACT**  
To alleviate the shortage of computing power faced by clients in training deep neural networks (DNNs) using federated learning (FL), we leverage the edge computing and split learning to propose a model-splitting allowed FL (SFL) framework, with the aim to minimize the training latency without loss of test accuracy. Under the synchronized global update setting, the latency to complete a round of global training is determined by the maximum latency for the clients to complete a local training session. Therefore, the training latency minimization problem (TLMP) is modelled as a minimizing-maximum problem. To solve this mixed integer nonlinear programming problem, we first propose a regression method to fit the quantitative-relationship between the cut-layer and other parameters of an AI-model, and thus, transform the TLMP into a continuous problem. Considering that the two subproblems involved in the TLMP, namely, the cut-layer selection problem for the clients and the computing resource allocation problem for the parameter-server are relative independence, an alternate-optimization-based algorithm with polynomial time complexity is developed to obtain a high-quality solution to the TLMP. Extensive experiments are performed on a popular DNN-model EfficientNetV2 using dataset MNIST, and the results verify the validity and improved performance of the proposed SFL framework.

{{</citation>}}


### (6/84) Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting (Marcel Kollovieh et al., 2023)

{{<citation>}}

Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, Yuyang Wang. (2023)  
**Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.11494v1)  

---


**ABSTRACT**  
Diffusion models have achieved state-of-the-art performance in generative modeling tasks across various domains. Prior works on time series diffusion models have primarily focused on developing conditional models tailored to specific forecasting or imputation tasks. In this work, we explore the potential of task-agnostic, unconditional diffusion models for several time series applications. We propose TSDiff, an unconditionally trained diffusion model for time series. Our proposed self-guidance mechanism enables conditioning TSDiff for downstream tasks during inference, without requiring auxiliary networks or altering the training procedure. We demonstrate the effectiveness of our method on three different time series tasks: forecasting, refinement, and synthetic data generation. First, we show that TSDiff is competitive with several task-specific conditional forecasting methods (predict). Second, we leverage the learned implicit probability density of TSDiff to iteratively refine the predictions of base forecasters with reduced computational overhead over reverse diffusion (refine). Notably, the generative performance of the model remains intact -- downstream forecasters trained on synthetic samples from TSDiff outperform forecasters that are trained on samples from other state-of-the-art generative time series models, occasionally even outperforming models trained on real data (synthesize).

{{</citation>}}


### (7/84) A New Deep State-Space Analysis Framework for Patient Latent State Estimation and Classification from EHR Time Series Data (Aya Nakamura et al., 2023)

{{<citation>}}

Aya Nakamura, Ryosuke Kojima, Yuji Okamoto, Eiichiro Uchino, Yohei Mineharu, Yohei Harada, Mayumi Kamada, Manabu Muto, Motoko Yanagita, Yasushi Okuno. (2023)  
**A New Deep State-Space Analysis Framework for Patient Latent State Estimation and Classification from EHR Time Series Data**  

---
Primary Category: cs.LG  
Categories: J-3; I-2-1, cs-LG, cs.LG  
Keywords: AI, Time Series  
[Paper Link](http://arxiv.org/abs/2307.11487v1)  

---


**ABSTRACT**  
Many diseases, including cancer and chronic conditions, require extended treatment periods and long-term strategies. Machine learning and AI research focusing on electronic health records (EHRs) have emerged to address this need. Effective treatment strategies involve more than capturing sequential changes in patient test values. It requires an explainable and clinically interpretable model by capturing the patient's internal state over time.   In this study, we propose the "deep state-space analysis framework," using time-series unsupervised learning of EHRs with a deep state-space model. This framework enables learning, visualizing, and clustering of temporal changes in patient latent states related to disease progression.   We evaluated our framework using time-series laboratory data from 12,695 cancer patients. By estimating latent states, we successfully discover latent states related to prognosis. By visualization and cluster analysis, the temporal transition of patient status and test items during state transitions characteristic of each anticancer drug were identified. Our framework surpasses existing methods in capturing interpretable latent space. It can be expected to enhance our comprehension of disease progression from EHRs, aiding treatment adjustments and prognostic determinations.

{{</citation>}}


### (8/84) A Deep Learning Approach for Overall Survival prediction in Lung Cancer with Missing Values (Camillo Maria Caruso et al., 2023)

{{<citation>}}

Camillo Maria Caruso, Valerio Guarrasi, Sara Ramella, Paolo Soda. (2023)  
**A Deep Learning Approach for Overall Survival prediction in Lung Cancer with Missing Values**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11465v2)  

---


**ABSTRACT**  
One of the most challenging fields where Artificial Intelligence (AI) can be applied is lung cancer research, specifically non-small cell lung cancer (NSCLC). In particular, overall survival (OS), the time between diagnosis and death, is a vital indicator of patient status, enabling tailored treatment and improved OS rates. In this analysis, there are two challenges to take into account. First, few studies effectively exploit the information available from each patient, leveraging both uncensored (i.e., dead) and censored (i.e., survivors) patients, considering also the events' time. Second, the handling of incomplete data is a common issue in the medical field. This problem is typically tackled through the use of imputation methods. Our objective is to present an AI model able to overcome these limits, effectively learning from both censored and uncensored patients and their available features, for the prediction of OS for NSCLC patients. We present a novel approach to survival analysis with missing values in the context of NSCLC, which exploits the strengths of the transformer architecture to account only for available features without requiring any imputation strategy. By making use of ad-hoc losses for OS, it is able to account for both censored and uncensored patients, as well as changes in risks over time. We compared our method with state-of-the-art models for survival analysis coupled with different imputation strategies. We evaluated the results obtained over a period of 6 years using different time granularities obtaining a Ct-index, a time-dependent variant of the C-index, of 71.97, 77.58 and 80.72 for time units of 1 month, 1 year and 2 years, respectively, outperforming all state-of-the-art methods regardless of the imputation method used.

{{</citation>}}


### (9/84) Batching for Green AI -- An Exploratory Study on Inference (Tim Yarally et al., 2023)

{{<citation>}}

Tim Yarally, Luís Cruz, Daniel Feitosa, June Sallou, Arie van Deursen. (2023)  
**Batching for Green AI -- An Exploratory Study on Inference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-SE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11434v1)  

---


**ABSTRACT**  
The batch size is an essential parameter to tune during the development of new neural networks. Amongst other quality indicators, it has a large degree of influence on the model's accuracy, generalisability, training times and parallelisability. This fact is generally known and commonly studied. However, during the application phase of a deep learning model, when the model is utilised by an end-user for inference, we find that there is a disregard for the potential benefits of introducing a batch size. In this study, we examine the effect of input batching on the energy consumption and response times of five fully-trained neural networks for computer vision that were considered state-of-the-art at the time of their publication. The results suggest that batching has a significant effect on both of these metrics. Furthermore, we present a timeline of the energy efficiency and accuracy of neural networks over the past decade. We find that in general, energy consumption rises at a much steeper pace than accuracy and question the necessity of this evolution. Additionally, we highlight one particular network, ShuffleNetV2(2018), that achieved a competitive performance for its time while maintaining a much lower energy consumption. Nevertheless, we highlight that the results are model dependent.

{{</citation>}}


### (10/84) Unsupervised Embedding Learning for Human Activity Recognition Using Wearable Sensor Data (Taoran Sheng et al., 2023)

{{<citation>}}

Taoran Sheng, Manfred Huber. (2023)  
**Unsupervised Embedding Learning for Human Activity Recognition Using Wearable Sensor Data**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG, eess-SP  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.11796v1)  

---


**ABSTRACT**  
The embedded sensors in widely used smartphones and other wearable devices make the data of human activities more accessible. However, recognizing different human activities from the wearable sensor data remains a challenging research problem in ubiquitous computing. One of the reasons is that the majority of the acquired data has no labels. In this paper, we present an unsupervised approach, which is based on the nature of human activity, to project the human activities into an embedding space in which similar activities will be located closely together. Using this, subsequent clustering algorithms can benefit from the embeddings, forming behavior clusters that represent the distinct activities performed by a person. Results of experiments on three labeled benchmark datasets demonstrate the effectiveness of the framework and show that our approach can help the clustering algorithm achieve improved performance in identifying and categorizing the underlying human activities compared to unsupervised techniques applied directly to the original data set.

{{</citation>}}


### (11/84) An Analysis of Multi-Agent Reinforcement Learning for Decentralized Inventory Control Systems (Marwan Mousa et al., 2023)

{{<citation>}}

Marwan Mousa, Damien van de Berg, Niki Kotecha, Ehecatl Antonio del Rio-Chanona, Max Mowbray. (2023)  
**An Analysis of Multi-Agent Reinforcement Learning for Decentralized Inventory Control Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11432v1)  

---


**ABSTRACT**  
Most solutions to the inventory management problem assume a centralization of information that is incompatible with organisational constraints in real supply chain networks. The inventory management problem is a well-known planning problem in operations research, concerned with finding the optimal re-order policy for nodes in a supply chain. While many centralized solutions to the problem exist, they are not applicable to real-world supply chains made up of independent entities. The problem can however be naturally decomposed into sub-problems, each associated with an independent entity, turning it into a multi-agent system. Therefore, a decentralized data-driven solution to inventory management problems using multi-agent reinforcement learning is proposed where each entity is controlled by an agent. Three multi-agent variations of the proximal policy optimization algorithm are investigated through simulations of different supply chain networks and levels of uncertainty. The centralized training decentralized execution framework is deployed, which relies on offline centralization during simulation-based policy identification, but enables decentralization when the policies are deployed online to the real system. Results show that using multi-agent proximal policy optimization with a centralized critic leads to performance very close to that of a centralized data-driven solution and outperforms a distributed model-based solution in most cases while respecting the information constraints of the system.

{{</citation>}}


### (12/84) Towards Better Fairness-Utility Trade-off: A Comprehensive Measurement-Based Reinforcement Learning Framework (Simiao Zhang et al., 2023)

{{<citation>}}

Simiao Zhang, Jitao Bai, Menghong Guan, Yihao Huang, Yueling Zhang, Jun Sun, Geguang Pu. (2023)  
**Towards Better Fairness-Utility Trade-off: A Comprehensive Measurement-Based Reinforcement Learning Framework**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11379v1)  

---


**ABSTRACT**  
Machine learning is widely used to make decisions with societal impact such as bank loan approving, criminal sentencing, and resume filtering. How to ensure its fairness while maintaining utility is a challenging but crucial issue. Fairness is a complex and context-dependent concept with over 70 different measurement metrics. Since existing regulations are often vague in terms of which metric to use and different organizations may prefer different fairness metrics, it is important to have means of improving fairness comprehensively. Existing mitigation techniques often target at one specific fairness metric and have limitations in improving multiple notions of fairness simultaneously. In this work, we propose CFU (Comprehensive Fairness-Utility), a reinforcement learning-based framework, to efficiently improve the fairness-utility trade-off in machine learning classifiers. A comprehensive measurement that can simultaneously consider multiple fairness notions as well as utility is established, and new metrics are proposed based on an in-depth analysis of the relationship between different fairness metrics. The reward function of CFU is constructed with comprehensive measurement and new metrics. We conduct extensive experiments to evaluate CFU on 6 tasks, 3 machine learning models, and 15 fairness-utility measurements. The results demonstrate that CFU can improve the classifier on multiple fairness metrics without sacrificing its utility. It outperforms all state-of-the-art techniques and has witnessed a 37.5% improvement on average.

{{</citation>}}


### (13/84) Bridging the Reality Gap of Reinforcement Learning based Traffic Signal Control using Domain Randomization and Meta Learning (Arthur Müller et al., 2023)

{{<citation>}}

Arthur Müller, Matthia Sabatelli. (2023)  
**Bridging the Reality Gap of Reinforcement Learning based Traffic Signal Control using Domain Randomization and Meta Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11357v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) has been widely explored in Traffic Signal Control (TSC) applications, however, still no such system has been deployed in practice. A key barrier to progress in this area is the reality gap, the discrepancy that results from differences between simulation models and their real-world equivalents. In this paper, we address this challenge by first presenting a comprehensive analysis of potential simulation parameters that contribute to this reality gap. We then also examine two promising strategies that can bridge this gap: Domain Randomization (DR) and Model-Agnostic Meta-Learning (MAML). Both strategies were trained with a traffic simulation model of an intersection. In addition, the model was embedded in LemgoRL, a framework that integrates realistic, safety-critical requirements into the control system. Subsequently, we evaluated the performance of the two methods on a separate model of the same intersection that was developed with a different traffic simulator. In this way, we mimic the reality gap. Our experimental results show that both DR and MAML outperform a state-of-the-art RL algorithm, therefore highlighting their potential to mitigate the reality gap in RLbased TSC systems.

{{</citation>}}


### (14/84) What can a Single Attention Layer Learn? A Study Through the Random Features Lens (Hengyu Fu et al., 2023)

{{<citation>}}

Hengyu Fu, Tianyu Guo, Yu Bai, Song Mei. (2023)  
**What can a Single Attention Layer Learn? A Study Through the Random Features Lens**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-ST, stat-ML, stat-TH  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.11353v1)  

---


**ABSTRACT**  
Attention layers -- which map a sequence of inputs to a sequence of outputs -- are core building blocks of the Transformer architecture which has achieved significant breakthroughs in modern artificial intelligence. This paper presents a rigorous theoretical study on the learning and generalization of a single multi-head attention layer, with a sequence of key vectors and a separate query vector as input. We consider the random feature setting where the attention layer has a large number of heads, with randomly sampled frozen query and key matrices, and trainable value matrices. We show that such a random-feature attention layer can express a broad class of target functions that are permutation invariant to the key vectors. We further provide quantitative excess risk bounds for learning these target functions from finite samples, using random feature attention with finitely many heads.   Our results feature several implications unique to the attention structure compared with existing random features theory for neural networks, such as (1) Advantages in the sample complexity over standard two-layer random-feature networks; (2) Concrete and natural classes of functions that can be learned efficiently by a random-feature attention layer; and (3) The effect of the sampling distribution of the query-key weight matrix (the product of the query and key matrix), where Gaussian random weights with a non-zero mean result in better sample complexities over the zero-mean counterpart for learning certain natural target functions. Experiments on simulated data corroborate our theoretical findings and further illustrate the interplay between the sample size and the complexity of the target function.

{{</citation>}}


### (15/84) Model-based Offline Reinforcement Learning with Count-based Conservatism (Byeongchan Kim et al., 2023)

{{<citation>}}

Byeongchan Kim, Min-hwan Oh. (2023)  
**Model-based Offline Reinforcement Learning with Count-based Conservatism**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11352v1)  

---


**ABSTRACT**  
In this paper, we propose a model-based offline reinforcement learning method that integrates count-based conservatism, named $\texttt{Count-MORL}$. Our method utilizes the count estimates of state-action pairs to quantify model estimation error, marking the first algorithm of demonstrating the efficacy of count-based conservatism in model-based offline deep RL to the best of our knowledge. For our proposed method, we first show that the estimation error is inversely proportional to the frequency of state-action pairs. Secondly, we demonstrate that the learned policy under the count-based conservative model offers near-optimality performance guarantees. Through extensive numerical experiments, we validate that $\texttt{Count-MORL}$ with hash code implementation significantly outperforms existing offline RL algorithms on the D4RL benchmark datasets. The code is accessible at $\href{https://github.com/oh-lab/Count-MORL}{https://github.com/oh-lab/Count-MORL}$.

{{</citation>}}


### (16/84) Improving Transferability of Adversarial Examples via Bayesian Attacks (Qizhang Li et al., 2023)

{{<citation>}}

Qizhang Li, Yiwen Guo, Xiaochen Yang, Wangmeng Zuo, Hao Chen. (2023)  
**Improving Transferability of Adversarial Examples via Bayesian Attacks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.11334v1)  

---


**ABSTRACT**  
This paper presents a substantial extension of our work published at ICLR. Our ICLR work advocated for enhancing transferability in adversarial examples by incorporating a Bayesian formulation into model parameters, which effectively emulates the ensemble of infinitely many deep neural networks, while, in this paper, we introduce a novel extension by incorporating the Bayesian formulation into the model input as well, enabling the joint diversification of both the model input and model parameters. Our empirical findings demonstrate that: 1) the combination of Bayesian formulations for both the model input and model parameters yields significant improvements in transferability; 2) by introducing advanced approximations of the posterior distribution over the model input, adversarial transferability achieves further enhancement, surpassing all state-of-the-arts when attacking without model fine-tuning. Moreover, we propose a principled approach to fine-tune model parameters in such an extended Bayesian formulation. The derived optimization objective inherently encourages flat minima in the parameter space and input space. Extensive experiments demonstrate that our method achieves a new state-of-the-art on transfer-based attacks, improving the average success rate on ImageNet and CIFAR-10 by 19.14% and 2.08%, respectively, when comparing with our ICLR basic Bayesian method. We will make our code publicly available.

{{</citation>}}


### (17/84) XLDA: Linear Discriminant Analysis for Scaling Continual Learning to Extreme Classification at the Edge (Karan Shah et al., 2023)

{{<citation>}}

Karan Shah, Vishruth Veerendranath, Anushka Hebbar, Raghavendra Bhat. (2023)  
**XLDA: Linear Discriminant Analysis for Scaling Continual Learning to Extreme Classification at the Edge**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2307.11317v1)  

---


**ABSTRACT**  
Streaming Linear Discriminant Analysis (LDA) while proven in Class-incremental Learning deployments at the edge with limited classes (upto 1000), has not been proven for deployment in extreme classification scenarios. In this paper, we present: (a) XLDA, a framework for Class-IL in edge deployment where LDA classifier is proven to be equivalent to FC layer including in extreme classification scenarios, and (b) optimizations to enable XLDA-based training and inference for edge deployment where there is a constraint on available compute resources. We show up to 42x speed up using a batched training approach and up to 5x inference speedup with nearest neighbor search on extreme datasets like AliProducts (50k classes) and Google Landmarks V2 (81k classes)

{{</citation>}}


### (18/84) PI-VEGAN: Physics Informed Variational Embedding Generative Adversarial Networks for Stochastic Differential Equations (Ruisong Gao et al., 2023)

{{<citation>}}

Ruisong Gao, Yufeng Wang, Min Yang, Chuanjun Chen. (2023)  
**PI-VEGAN: Physics Informed Variational Embedding Generative Adversarial Networks for Stochastic Differential Equations**  

---
Primary Category: cs.LG  
Categories: 65Yxx, G-0, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.11289v1)  

---


**ABSTRACT**  
We present a new category of physics-informed neural networks called physics informed variational embedding generative adversarial network (PI-VEGAN), that effectively tackles the forward, inverse, and mixed problems of stochastic differential equations. In these scenarios, the governing equations are known, but only a limited number of sensor measurements of the system parameters are available. We integrate the governing physical laws into PI-VEGAN with automatic differentiation, while introducing a variational encoder for approximating the latent variables of the actual distribution of the measurements. These latent variables are integrated into the generator to facilitate accurate learning of the characteristics of the stochastic partial equations. Our model consists of three components, namely the encoder, generator, and discriminator, each of which is updated alternatively employing the stochastic gradient descent algorithm. We evaluate the effectiveness of PI-VEGAN in addressing forward, inverse, and mixed problems that require the concurrent calculation of system parameters and solutions. Numerical results demonstrate that the proposed method achieves satisfactory stability and accuracy in comparison with the previous physics-informed generative adversarial network (PI-WGAN).

{{</citation>}}


## cs.DL (1)



### (19/84) Bibliometric Analysis of Publisher and Journal Instructions to Authors on Generative-AI in Academic and Scientific Publishing (Conner Ganjavi et al., 2023)

{{<citation>}}

Conner Ganjavi, Michael B. Eppler, Asli Pekcan, Brett Biedermann, Andre Abreu, Gary S. Collins, Inderbir S. Gill, Giovanni E. Cacciamani. (2023)  
**Bibliometric Analysis of Publisher and Journal Instructions to Authors on Generative-AI in Academic and Scientific Publishing**  

---
Primary Category: cs.DL  
Categories: A-0, cs-AI, cs-DL, cs.DL  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.11918v1)  

---


**ABSTRACT**  
We aim to determine the extent and content of guidance for authors regarding the use of generative-AI (GAI), Generative Pretrained models (GPTs) and Large Language Models (LLMs) powered tools among the top 100 academic publishers and journals in science. The websites of these publishers and journals were screened from between 19th and 20th May 2023. Among the largest 100 publishers, 17% provided guidance on the use of GAI, of which 12 (70.6%) were among the top 25 publishers. Among the top 100 journals, 70% have provided guidance on GAI. Of those with guidance, 94.1% of publishers and 95.7% of journals prohibited the inclusion of GAI as an author. Four journals (5.7%) explicitly prohibit the use of GAI in the generation of a manuscript, while 3 (17.6%) publishers and 15 (21.4%) journals indicated their guidance exclusively applies to the writing process. When disclosing the use of GAI, 42.8% of publishers and 44.3% of journals included specific disclosure criteria. There was variability in guidance of where to disclose the use of GAI, including in the methods, acknowledgments, cover letter, or a new section. There was also variability in how to access GAI guidance and the linking of journal and publisher instructions to authors. There is a lack of guidance by some top publishers and journals on the use of GAI by authors. Among those publishers and journals that provide guidance, there is substantial heterogeneity in the allowable uses of GAI and in how it should be disclosed, with this heterogeneity persisting among affiliated publishers and journals in some instances. The lack of standardization burdens authors and threatens to limit the effectiveness of these regulations. There is a need for standardized guidelines in order to protect the integrity of scientific output as GAI continues to grow in popularity.

{{</citation>}}


## cs.SE (3)



### (20/84) Vulnerability Detection Through an Adversarial Fuzzing Algorithm (Michael Wang et al., 2023)

{{<citation>}}

Michael Wang, Michael Robinson. (2023)  
**Vulnerability Detection Through an Adversarial Fuzzing Algorithm**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2307.11917v1)  

---


**ABSTRACT**  
Fuzzing is a popular vulnerability automated testing method utilized by professionals and broader community alike. However, despite its abilities, fuzzing is a time-consuming, computationally expensive process. This is problematic for the open source community and smaller developers, as most people will not have dedicated security professionals and/or knowledge to perform extensive testing on their own. The goal of this project is to increase the efficiency of existing fuzzers by allowing fuzzers to explore more paths and find more bugs in shorter amounts of time, while still remaining operable on a personal device. To accomplish this, adversarial methods are built on top of current evolutionary algorithms to generate test cases for further and more efficient fuzzing. The results of this show that adversarial attacks do in fact increase outpaces existing fuzzers significantly and, consequently, crashes found.

{{</citation>}}


### (21/84) Exploring Technical Debt in Security Questions on Stack Overflow (Joshua Aldrich Edbert et al., 2023)

{{<citation>}}

Joshua Aldrich Edbert, Sahrima Jannat Oishwee, Shubhashis Karmakar, Zadia Codabux, Roberto Verdecchia. (2023)  
**Exploring Technical Debt in Security Questions on Stack Overflow**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.11387v1)  

---


**ABSTRACT**  
Background: Software security is crucial to ensure that the users are protected from undesirable consequences such as malware attacks which can result in loss of data and, subsequently, financial loss. Technical Debt (TD) is a metaphor incurred by suboptimal decisions resulting in long-term consequences such as increased defects and vulnerabilities if not managed. Although previous studies have studied the relationship between security and TD, examining their intersection in developers' discussion on Stack Overflow (SO) is still unexplored. Aims: This study investigates the characteristics of security-related TD questions on SO. More specifically, we explore the prevalence of TD in security-related queries, identify the security tags most prone to TD, and investigate which user groups are more aware of TD. Method: We mined 117,233 security-related questions on SO and used a deep-learning approach to identify 45,078 security-related TD questions. Subsequently, we conducted quantitative and qualitative analyses of the collected security-related TD questions, including sentiment analysis. Results: Our analysis revealed that 38% of the security questions on SO are security-related TD questions. The most recurrent tags among the security-related TD questions emerged as "security" and "encryption." The latter typically have a neutral sentiment, are lengthier, and are posed by users with higher reputation scores. Conclusions: Our findings reveal that developers implicitly discuss TD, suggesting developers have a potential knowledge gap regarding the TD metaphor in the security domain. Moreover, we identified the most common security topics mentioned in TD-related posts, providing valuable insights for developers and researchers to assist developers in prioritizing security concerns in order to minimize TD and enhance software security.

{{</citation>}}


### (22/84) DEFTri: A Few-Shot Label Fused Contextual Representation Learning For Product Defect Triage in e-Commerce (Ipsita Mohanty, 2023)

{{<citation>}}

Ipsita Mohanty. (2023)  
**DEFTri: A Few-Shot Label Fused Contextual Representation Learning For Product Defect Triage in e-Commerce**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: BERT, Few-Shot, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.11344v1)  

---


**ABSTRACT**  
Defect Triage is a time-sensitive and critical process in a large-scale agile software development lifecycle for e-commerce. Inefficiencies arising from human and process dependencies in this domain have motivated research in automated approaches using machine learning to accurately assign defects to qualified teams. This work proposes a novel framework for automated defect triage (DEFTri) using fine-tuned state-of-the-art pre-trained BERT on labels fused text embeddings to improve contextual representations from human-generated product defects. For our multi-label text classification defect triage task, we also introduce a Walmart proprietary dataset of product defects using weak supervision and adversarial learning, in a few-shot setting.

{{</citation>}}


## cs.NI (1)



### (23/84) Software defined networking flow admission and routing under minimal security constraints (Jorge López et al., 2023)

{{<citation>}}

Jorge López, Charalampos Chatzinakis, Marc Cartigny, Claude Poletti. (2023)  
**Software defined networking flow admission and routing under minimal security constraints**  

---
Primary Category: cs.NI  
Categories: cs-CR, cs-NI, cs.NI  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.11879v1)  

---


**ABSTRACT**  
In recent years, computer networks and telecommunications in general have been shifting paradigms to adopt software-centric approaches. Software Defined Networking (SDN) is one of such paradigms that centralizes control and intelligent applications can be defined on top of this architecture. The latter enables the definition of the network behavior by means of software. In this work, we propose an approach for Flow Admission and Routing under Minimal Security Constraints (FARSec) in Software Defined Networks, where network flows must use links which are at least as secure as their required security level. We prove that FARSec can find feasible paths that respect the minimum level of security for each flow. If the latter is not possible FARSec rejects the flow in order not to compromise its security. We show that the computational complexity of the proposed approach is polynomial. Experimental results with semi-random generated graphs confirm the efficiency and correctness of the proposed approach. Finally, we implement the proposed solution using OpenFlow and ONOS -- an SDN open-source controller. We validate its functionality using an emulated network with various security levels.

{{</citation>}}


## cs.RO (5)



### (24/84) CARTIER: Cartographic lAnguage Reasoning Targeted at Instruction Execution for Robots (Nikhil Kakodkar et al., 2023)

{{<citation>}}

Nikhil Kakodkar, Dmitriy Rivkin, Bobak H. Baghi, Francois Hogan, Gregory Dudek. (2023)  
**CARTIER: Cartographic lAnguage Reasoning Targeted at Instruction Execution for Robots**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-RO, cs.RO  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.11865v1)  

---


**ABSTRACT**  
This work explores the capacity of large language models (LLMs) to address problems at the intersection of spatial planning and natural language interfaces for navigation.Our focus is on following relatively complex instructions that are more akin to natural conversation than traditional explicit procedural directives seen in robotics. Unlike most prior work, where navigation directives are provided as imperative commands (e.g., go to the fridge), we examine implicit directives within conversational interactions. We leverage the 3D simulator AI2Thor to create complex and repeatable scenarios at scale, and augment it by adding complex language queries for 40 object types. We demonstrate that a robot can better parse descriptive language queries than existing methods by using an LLM to interpret the user interaction in the context of a list of the objects in the scene.

{{</citation>}}


### (25/84) 3D Skeletonization of Complex Grapevines for Robotic Pruning (Eric Schneider et al., 2023)

{{<citation>}}

Eric Schneider, Sushanth Jayanth, Abhisesh Silwal, George Kantor. (2023)  
**3D Skeletonization of Complex Grapevines for Robotic Pruning**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2307.11706v1)  

---


**ABSTRACT**  
Robotic pruning of dormant grapevines is an area of active research in order to promote vine balance and grape quality, but so far robotic efforts have largely focused on planar, simplified vines not representative of commercial vineyards. This paper aims to advance the robotic perception capabilities necessary for pruning in denser and more complex vine structures by extending plant skeletonization techniques. The proposed pipeline generates skeletal grapevine models that have lower reprojection error and higher connectivity than baseline algorithms. We also show how 3D and skeletal information enables prediction accuracy of pruning weight for dense vines surpassing prior work, where pruning weight is an important vine metric influencing pruning site selection.

{{</citation>}}


### (26/84) BatMobility: Towards Flying Without Seeing for Autonomous Drones (Emerson Sie et al., 2023)

{{<citation>}}

Emerson Sie, Zikun Liu, Deepak Vasisht. (2023)  
**BatMobility: Towards Flying Without Seeing for Autonomous Drones**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.11518v1)  

---


**ABSTRACT**  
Unmanned aerial vehicles (UAVs) rely on optical sensors such as cameras and lidar for autonomous operation. However, such optical sensors are error-prone in bad lighting, inclement weather conditions including fog and smoke, and around textureless or transparent surfaces. In this paper, we ask: is it possible to fly UAVs without relying on optical sensors, i.e., can UAVs fly without seeing? We present BatMobility, a lightweight mmWave radar-only perception system for UAVs that eliminates the need for optical sensors. BatMobility enables two core functionalities for UAVs -- radio flow estimation (a novel FMCW radar-based alternative for optical flow based on surface-parallel doppler shift) and radar-based collision avoidance. We build BatMobility using commodity sensors and deploy it as a real-time system on a small off-the-shelf quadcopter running an unmodified flight controller. Our evaluation shows that BatMobility achieves comparable or better performance than commercial-grade optical sensors across a wide range of scenarios.

{{</citation>}}


### (27/84) EV-Planner: Energy-Efficient Robot Navigation via Event-Based Physics-Guided Neuromorphic Planner (Sourav Sanyal et al., 2023)

{{<citation>}}

Sourav Sanyal, Rohan Kumar Manna, Kaushik Roy. (2023)  
**EV-Planner: Energy-Efficient Robot Navigation via Event-Based Physics-Guided Neuromorphic Planner**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11349v1)  

---


**ABSTRACT**  
Vision-based object tracking is an essential precursor to performing autonomous aerial navigation in order to avoid obstacles. Biologically inspired neuromorphic event cameras are emerging as a powerful alternative to frame-based cameras, due to their ability to asynchronously detect varying intensities (even in poor lighting conditions), high dynamic range, and robustness to motion blur. Spiking neural networks (SNNs) have gained traction for processing events asynchronously in an energy-efficient manner. On the other hand, physics-based artificial intelligence (AI) has gained prominence recently, as they enable embedding system knowledge via physical modeling inside traditional analog neural networks (ANNs). In this letter, we present an event-based physics-guided neuromorphic planner (EV-Planner) to perform obstacle avoidance using neuromorphic event cameras and physics-based AI. We consider the task of autonomous drone navigation where the mission is to detect moving gates and fly through them while avoiding a collision. We use event cameras to perform object detection using a shallow spiking neural network in an unsupervised fashion. Utilizing the physical equations of the brushless DC motors present in the drone rotors, we train a lightweight energy-aware physics-guided neural network with depth inputs. This predicts the optimal flight time responsible for generating near-minimum energy paths. We spawn the drone in the Gazebo simulator and implement a sensor-fused vision-to-planning neuro-symbolic framework using Robot Operating System (ROS). Simulation results for safe collision-free flight trajectories are presented with performance analysis and potential future research directions

{{</citation>}}


### (28/84) How to Tidy Up a Table: Fusing Visual and Semantic Commonsense Reasoning for Robotic Tasks with Vague Objectives (Yiqing Xu et al., 2023)

{{<citation>}}

Yiqing Xu, David Hsu. (2023)  
**How to Tidy Up a Table: Fusing Visual and Semantic Commonsense Reasoning for Robotic Tasks with Vague Objectives**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.11319v1)  

---


**ABSTRACT**  
Vague objectives in many real-life scenarios pose long-standing challenges for robotics, as defining rules, rewards, or constraints for optimization is difficult. Tasks like tidying a messy table may appear simple for humans, but articulating the criteria for tidiness is complex due to the ambiguity and flexibility in commonsense reasoning. Recent advancement in Large Language Models (LLMs) offers us an opportunity to reason over these vague objectives: learned from extensive human data, LLMs capture meaningful common sense about human behavior. However, as LLMs are trained solely on language input, they may struggle with robotic tasks due to their limited capacity to account for perception and low-level controls. In this work, we propose a simple approach to solve the task of table tidying, an example of robotic tasks with vague objectives. Specifically, the task of tidying a table involves not just clustering objects by type and functionality for semantic tidiness but also considering spatial-visual relations of objects for a visually pleasing arrangement, termed as visual tidiness. We propose to learn a lightweight, image-based tidiness score function to ground the semantically tidy policy of LLMs to achieve visual tidiness. We innovatively train the tidiness score using synthetic data gathered using random walks from a few tidy configurations. Such trajectories naturally encode the order of tidiness, thereby eliminating the need for laborious and expensive human demonstrations. Our empirical results show that our pipeline can be applied to unseen objects and complex 3D arrangements.

{{</citation>}}


## cs.SI (4)



### (29/84) The Looming Threat of Fake and LLM-generated LinkedIn Profiles: Challenges and Opportunities for Detection and Prevention (Navid Ayoobi et al., 2023)

{{<citation>}}

Navid Ayoobi, Sadat Shahriar, Arjun Mukherjee. (2023)  
**The Looming Threat of Fake and LLM-generated LinkedIn Profiles: Challenges and Opportunities for Detection and Prevention**  

---
Primary Category: cs.SI  
Categories: cs-CL, cs-CR, cs-LG, cs-SI, cs.SI  
Keywords: BERT, Embedding, Language Model, Social Network  
[Paper Link](http://arxiv.org/abs/2307.11864v1)  

---


**ABSTRACT**  
In this paper, we present a novel method for detecting fake and Large Language Model (LLM)-generated profiles in the LinkedIn Online Social Network immediately upon registration and before establishing connections. Early fake profile identification is crucial to maintaining the platform's integrity since it prevents imposters from acquiring the private and sensitive information of legitimate users and from gaining an opportunity to increase their credibility for future phishing and scamming activities. This work uses textual information provided in LinkedIn profiles and introduces the Section and Subsection Tag Embedding (SSTE) method to enhance the discriminative characteristics of these data for distinguishing between legitimate profiles and those created by imposters manually or by using an LLM. Additionally, the dearth of a large publicly available LinkedIn dataset motivated us to collect 3600 LinkedIn profiles for our research. We will release our dataset publicly for research purposes. This is, to the best of our knowledge, the first large publicly available LinkedIn dataset for fake LinkedIn account detection. Within our paradigm, we assess static and contextualized word embeddings, including GloVe, Flair, BERT, and RoBERTa. We show that the suggested method can distinguish between legitimate and fake profiles with an accuracy of about 95% across all word embeddings. In addition, we show that SSTE has a promising accuracy for identifying LLM-generated profiles, despite the fact that no LLM-generated profiles were employed during the training phase, and can achieve an accuracy of approximately 90% when only 20 LLM-generated profiles are added to the training set. It is a significant finding since the proliferation of several LLMs in the near future makes it extremely challenging to design a single system that can identify profiles created with various LLMs.

{{</citation>}}


### (30/84) Diurnal Patterns in the Spread of COVID-19 Misinformation on Twitter within Italy (Elisabeth Stockinger et al., 2023)

{{<citation>}}

Elisabeth Stockinger, Riccardo Gallotti, Carina I. Hausladen. (2023)  
**Diurnal Patterns in the Spread of COVID-19 Misinformation on Twitter within Italy**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.11575v1)  

---


**ABSTRACT**  
Social media manipulation poses a significant threat to cognitive autonomy and unbiased opinion formation. Prior literature explored the relationship between online activity, and emotional state, cognitive resources, sunlight, and weather. However, a limited understanding exists regarding the role of time of day in content spread and the impact of user activity patterns and chronotype on susceptibility to mis- and disinformation. This work uncovers a strong correlation between user activity patterns and the tendency to spread manipulated content. Through quantitative analysis of Twitter data, we examine how user activity throughout the day aligns with chronotypical archetypes. Evening types exhibit a significantly higher inclination towards spreading potentially manipulated content, which is generally more likely between 2:30 AM and 4:15 AM. This knowledge can become crucial for developing targeted interventions and strategies that mitigate misinformation spread by addressing vulnerable periods and user groups more susceptible to manipulation.

{{</citation>}}


### (31/84) Prompt-Based Zero- and Few-Shot Node Classification: A Multimodal Approach (Yuexin Li et al., 2023)

{{<citation>}}

Yuexin Li, Bryan Hooi. (2023)  
**Prompt-Based Zero- and Few-Shot Node Classification: A Multimodal Approach**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.11572v1)  

---


**ABSTRACT**  
Multimodal data empowers machine learning models to better understand the world from various perspectives. In this work, we study the combination of \emph{text and graph} modalities, a challenging but understudied combination which is prevalent across multiple settings including citation networks, social media, and the web. We focus on the popular task of node classification using limited labels; in particular, under the zero- and few-shot scenarios. In contrast to the standard pipeline which feeds standard precomputed (e.g., bag-of-words) text features into a graph neural network, we propose \textbf{T}ext-\textbf{A}nd-\textbf{G}raph (TAG) learning, a more deeply multimodal approach that integrates the raw texts and graph topology into the model design, and can effectively learn from limited supervised signals without any meta-learning procedure. TAG is a two-stage model with (1) a prompt- and graph-based module which generates prior logits that can be directly used for zero-shot node classification, and (2) a trainable module that further calibrates these prior logits in a few-shot manner. Experiments on two node classification datasets show that TAG outperforms all the baselines by a large margin in both zero- and few-shot settings.

{{</citation>}}


### (32/84) Friction Interventions to Curb the Spread of Misinformation on Social Media (Laura Jahn et al., 2023)

{{<citation>}}

Laura Jahn, Rasmus K. Rendsvig, Alessandro Flammini, Filippo Menczer, Vincent F. Hendricks. (2023)  
**Friction Interventions to Curb the Spread of Misinformation on Social Media**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2307.11498v1)  

---


**ABSTRACT**  
Social media has enabled the spread of information at unprecedented speeds and scales, and with it the proliferation of high-engagement, low-quality content. *Friction* -- behavioral design measures that make the sharing of content more cumbersome -- might be a way to raise the quality of what is spread online. Here, we study the effects of friction with and without quality-recognition learning. Experiments from an agent-based model suggest that friction alone decreases the number of posts without improving their quality. A small amount of friction combined with learning, however, increases the average quality of posts significantly. Based on this preliminary evidence, we propose a friction intervention with a learning component about the platform's community standards, to be tested via a field experiment. The proposed intervention would have minimal effects on engagement and may easily be deployed at scale.

{{</citation>}}


## cs.CV (23)



### (33/84) Digital Modeling on Large Kernel Metamaterial Neural Network (Quan Liu et al., 2023)

{{<citation>}}

Quan Liu, Hanyu Zheng, Brandon T. Swartz, Ho hin Lee, Zuhayr Asad, Ivan Kravchenko, Jason G. Valentine, Yuankai Huo. (2023)  
**Digital Modeling on Large Kernel Metamaterial Neural Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11862v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) utilized recently are physically deployed with computational units (e.g., CPUs and GPUs). Such a design might lead to a heavy computational burden, significant latency, and intensive power consumption, which are critical limitations in applications such as the Internet of Things (IoT), edge computing, and the usage of drones. Recent advances in optical computational units (e.g., metamaterial) have shed light on energy-free and light-speed neural networks. However, the digital design of the metamaterial neural network (MNN) is fundamentally limited by its physical limitations, such as precision, noise, and bandwidth during fabrication. Moreover, the unique advantages of MNN's (e.g., light-speed computation) are not fully explored via standard 3x3 convolution kernels. In this paper, we propose a novel large kernel metamaterial neural network (LMNN) that maximizes the digital capacity of the state-of-the-art (SOTA) MNN with model re-parametrization and network compression, while also considering the optical limitation explicitly. The new digital learning scheme can maximize the learning capacity of MNN while modeling the physical restrictions of meta-optic. With the proposed LMNN, the computation cost of the convolutional front-end can be offloaded into fabricated optical hardware. The experimental results on two publicly available datasets demonstrate that the optimized hybrid design improved classification accuracy while reducing computational latency. The development of the proposed LMNN is a promising step towards the ultimate goal of energy-free and light-speed AI.

{{</citation>}}


### (34/84) HybridAugment++: Unified Frequency Spectra Perturbations for Model Robustness (Mehmet Kerim Yucel et al., 2023)

{{<citation>}}

Mehmet Kerim Yucel, Ramazan Gokberk Cinbis, Pinar Duygulu. (2023)  
**HybridAugment++: Unified Frequency Spectra Perturbations for Model Robustness**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.11823v1)  

---


**ABSTRACT**  
Convolutional Neural Networks (CNN) are known to exhibit poor generalization performance under distribution shifts. Their generalization have been studied extensively, and one line of work approaches the problem from a frequency-centric perspective. These studies highlight the fact that humans and CNNs might focus on different frequency components of an image. First, inspired by these observations, we propose a simple yet effective data augmentation method HybridAugment that reduces the reliance of CNNs on high-frequency components, and thus improves their robustness while keeping their clean accuracy high. Second, we propose HybridAugment++, which is a hierarchical augmentation method that attempts to unify various frequency-spectrum augmentations. HybridAugment++ builds on HybridAugment, and also reduces the reliance of CNNs on the amplitude component of images, and promotes phase information instead. This unification results in competitive to or better than state-of-the-art results on clean accuracy (CIFAR-10/100 and ImageNet), corruption benchmarks (ImageNet-C, CIFAR-10-C and CIFAR-100-C), adversarial robustness on CIFAR-10 and out-of-distribution detection on various datasets. HybridAugment and HybridAugment++ are implemented in a few lines of code, does not require extra data, ensemble models or additional networks.

{{</citation>}}


### (35/84) BandRe: Rethinking Band-Pass Filters for Scale-Wise Object Detection Evaluation (Yosuke Shinya, 2023)

{{<citation>}}

Yosuke Shinya. (2023)  
**BandRe: Rethinking Band-Pass Filters for Scale-Wise Object Detection Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.11748v1)  

---


**ABSTRACT**  
Scale-wise evaluation of object detectors is important for real-world applications. However, existing metrics are either coarse or not sufficiently reliable. In this paper, we propose novel scale-wise metrics that strike a balance between fineness and reliability, using a filter bank consisting of triangular and trapezoidal band-pass filters. We conduct experiments with two methods on two datasets and show that the proposed metrics can highlight the differences between the methods and between the datasets. Code is available at https://github.com/shinya7y/UniverseNet .

{{</citation>}}


### (36/84) Automatic Data Augmentation Learning using Bilevel Optimization for Histopathological Images (Saypraseuth Mounsaveng et al., 2023)

{{<citation>}}

Saypraseuth Mounsaveng, Issam Laradji, David Vázquez, Marco Perdersoli, Ismail Ben Ayed. (2023)  
**Automatic Data Augmentation Learning using Bilevel Optimization for Histopathological Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.11808v1)  

---


**ABSTRACT**  
Training a deep learning model to classify histopathological images is challenging, because of the color and shape variability of the cells and tissues, and the reduced amount of available data, which does not allow proper learning of those variations. Variations can come from the image acquisition process, for example, due to different cell staining protocols or tissue deformation. To tackle this challenge, Data Augmentation (DA) can be used during training to generate additional samples by applying transformations to existing ones, to help the model become invariant to those color and shape transformations. The problem with DA is that it is not only dataset-specific but it also requires domain knowledge, which is not always available. Without this knowledge, selecting the right transformations can only be done using heuristics or through a computationally demanding search. To address this, we propose an automatic DA learning method. In this method, the DA parameters, i.e. the transformation parameters needed to improve the model training, are considered learnable and are learned automatically using a bilevel optimization approach in a quick and efficient way using truncated backpropagation. We validated the method on six different datasets. Experimental results show that our model can learn color and affine transformations that are more helpful to train an image classifier than predefined DA transformations, which are also more expensive as they need to be selected before the training by grid search on a validation set. We also show that similarly to a model trained with RandAugment, our model has also only a few method-specific hyperparameters to tune but is performing better. This makes our model a good solution for learning the best DA parameters, especially in the context of histopathological images, where defining potentially useful transformation heuristically is not trivial.

{{</citation>}}


### (37/84) Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts (Mayug Maniparambil et al., 2023)

{{<citation>}}

Mayug Maniparambil, Chris Vorster, Derek Molloy, Noel Murphy, Kevin McGuinness, Noel E. O'Connor. (2023)  
**Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.11661v1)  

---


**ABSTRACT**  
Contrastive pretrained large Vision-Language Models (VLMs) like CLIP have revolutionized visual representation learning by providing good performance on downstream datasets. VLMs are 0-shot adapted to a downstream dataset by designing prompts that are relevant to the dataset. Such prompt engineering makes use of domain expertise and a validation dataset. Meanwhile, recent developments in generative pretrained models like GPT-4 mean they can be used as advanced internet search tools. They can also be manipulated to provide visual information in any structure. In this work, we show that GPT-4 can be used to generate text that is visually descriptive and how this can be used to adapt CLIP to downstream tasks. We show considerable improvements in 0-shot transfer accuracy on specialized fine-grained datasets like EuroSAT (~7%), DTD (~7%), SUN397 (~4.6%), and CUB (~3.3%) when compared to CLIP's default prompt. We also design a simple few-shot adapter that learns to choose the best possible sentences to construct generalizable classifiers that outperform the recently proposed CoCoOP by ~2% on average and by over 4% on 4 specialized fine-grained datasets. We will release the code, prompts, and auxiliary text dataset upon acceptance.

{{</citation>}}


### (38/84) Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detection and Classification Models (Jiajun Zhang et al., 2023)

{{<citation>}}

Jiajun Zhang, Georgina Cosma, Sarah Bugby, Axel Finke, Jason Watkins. (2023)  
**Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detection and Classification Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.11643v2)  

---


**ABSTRACT**  
As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind their predictions. This paper proposes the AI-Reasoner, which extracts the morphological characteristics of defects (DefChars) from images and utilises decision trees to reason with the DefChar values. Thereafter, the AI-Reasoner exports visualisations (i.e. charts) and textual explanations to provide insights into outputs made by masked-based defect detection and classification models. It also provides effective mitigation strategies to enhance data pre-processing and overall model performance. The AI-Reasoner was tested on explaining the outputs of an IE Mask R-CNN model using a set of 366 images containing defects. The results demonstrated its effectiveness in explaining the IE Mask R-CNN model's predictions. Overall, the proposed AI-Reasoner provides a solution for improving the performance of AI models in industrial applications that require defect analysis.

{{</citation>}}


### (39/84) Deep Reinforcement Learning Based System for Intraoperative Hyperspectral Video Autofocusing (Charlie Budd et al., 2023)

{{<citation>}}

Charlie Budd, Jianrong Qiu, Oscar MacCormac, Martin Huber, Christopher Mower, Mirek Janatka, Théo Trotouin, Jonathan Shapey, Mads S. Bergholt, Tom Vercauteren. (2023)  
**Deep Reinforcement Learning Based System for Intraoperative Hyperspectral Video Autofocusing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11638v1)  

---


**ABSTRACT**  
Hyperspectral imaging (HSI) captures a greater level of spectral detail than traditional optical imaging, making it a potentially valuable intraoperative tool when precise tissue differentiation is essential. Hardware limitations of current optical systems used for handheld real-time video HSI result in a limited focal depth, thereby posing usability issues for integration of the technology into the operating room. This work integrates a focus-tunable liquid lens into a video HSI exoscope, and proposes novel video autofocusing methods based on deep reinforcement learning. A first-of-its-kind robotic focal-time scan was performed to create a realistic and reproducible testing dataset. We benchmarked our proposed autofocus algorithm against traditional policies, and found our novel approach to perform significantly ($p<0.05$) better than traditional techniques ($0.070\pm.098$ mean absolute focal error compared to $0.146\pm.148$). In addition, we performed a blinded usability trial by having two neurosurgeons compare the system with different autofocus policies, and found our novel approach to be the most favourable, making our system a desirable addition for intraoperative HSI.

{{</citation>}}


### (40/84) Consistency-guided Meta-Learning for Bootstrapping Semi-Supervised Medical Image Segmentation (Qingyue Wei et al., 2023)

{{<citation>}}

Qingyue Wei, Lequan Yu, Xianhang Li, Wei Shao, Cihang Xie, Lei Xing, Yuyin Zhou. (2023)  
**Consistency-guided Meta-Learning for Bootstrapping Semi-Supervised Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.11604v1)  

---


**ABSTRACT**  
Medical imaging has witnessed remarkable progress but usually requires a large amount of high-quality annotated data which is time-consuming and costly to obtain. To alleviate this burden, semi-supervised learning has garnered attention as a potential solution. In this paper, we present Meta-Learning for Bootstrapping Medical Image Segmentation (MLB-Seg), a novel method for tackling the challenge of semi-supervised medical image segmentation. Specifically, our approach first involves training a segmentation model on a small set of clean labeled images to generate initial labels for unlabeled data. To further optimize this bootstrapping process, we introduce a per-pixel weight mapping system that dynamically assigns weights to both the initialized labels and the model's own predictions. These weights are determined using a meta-process that prioritizes pixels with loss gradient directions closer to those of clean data, which is based on a small set of precisely annotated images. To facilitate the meta-learning process, we additionally introduce a consistency-based Pseudo Label Enhancement (PLE) scheme that improves the quality of the model's own predictions by ensembling predictions from various augmented versions of the same input. In order to improve the quality of the weight maps obtained through multiple augmentations of a single input, we introduce a mean teacher into the PLE scheme. This method helps to reduce noise in the weight maps and stabilize its generation process. Our extensive experimental results on public atrial and prostate segmentation datasets demonstrate that our proposed method achieves state-of-the-art results under semi-supervision. Our code is available at https://github.com/aijinrjinr/MLB-Seg.

{{</citation>}}


### (41/84) Advancing Visual Grounding with Scene Knowledge: Benchmark and Method (Zhihong Chen et al., 2023)

{{<citation>}}

Zhihong Chen, Ruifei Zhang, Yibing Song, Xiang Wan, Guanbin Li. (2023)  
**Advancing Visual Grounding with Scene Knowledge: Benchmark and Method**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.11558v1)  

---


**ABSTRACT**  
Visual grounding (VG) aims to establish fine-grained alignment between vision and language. Ideally, it can be a testbed for vision-and-language models to evaluate their understanding of the images and texts and their reasoning abilities over their joint space. However, most existing VG datasets are constructed using simple description texts, which do not require sufficient reasoning over the images and texts. This has been demonstrated in a recent study~\cite{luo2022goes}, where a simple LSTM-based text encoder without pretraining can achieve state-of-the-art performance on mainstream VG datasets. Therefore, in this paper, we propose a novel benchmark of \underline{S}cene \underline{K}nowledge-guided \underline{V}isual \underline{G}rounding (SK-VG), where the image content and referring expressions are not sufficient to ground the target objects, forcing the models to have a reasoning ability on the long-form scene knowledge. To perform this task, we propose two approaches to accept the triple-type input, where the former embeds knowledge into the image features before the image-query interaction; the latter leverages linguistic structure to assist in computing the image-text matching. We conduct extensive experiments to analyze the above methods and show that the proposed approaches achieve promising results but still leave room for improvement, including performance and interpretability. The dataset and code are available at \url{https://github.com/zhjohnchan/SK-VG}.

{{</citation>}}


### (42/84) YOLOPose V2: Understanding and Improving Transformer-based 6D Pose Estimation (Arul Selvam Periyasamy et al., 2023)

{{<citation>}}

Arul Selvam Periyasamy, Arash Amini, Vladimir Tsaturyan, Sven Behnke. (2023)  
**YOLOPose V2: Understanding and Improving Transformer-based 6D Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.11550v1)  

---


**ABSTRACT**  
6D object pose estimation is a crucial prerequisite for autonomous robot manipulation applications. The state-of-the-art models for pose estimation are convolutional neural network (CNN)-based. Lately, Transformers, an architecture originally proposed for natural language processing, is achieving state-of-the-art results in many computer vision tasks as well. Equipped with the multi-head self-attention mechanism, Transformers enable simple single-stage end-to-end architectures for learning object detection and 6D object pose estimation jointly. In this work, we propose YOLOPose (short form for You Only Look Once Pose estimation), a Transformer-based multi-object 6D pose estimation method based on keypoint regression and an improved variant of the YOLOPose model. In contrast to the standard heatmaps for predicting keypoints in an image, we directly regress the keypoints. Additionally, we employ a learnable orientation estimation module to predict the orientation from the keypoints. Along with a separate translation estimation module, our model is end-to-end differentiable. Our method is suitable for real-time applications and achieves results comparable to state-of-the-art methods. We analyze the role of object queries in our architecture and reveal that the object queries specialize in detecting objects in specific image regions. Furthermore, we quantify the accuracy trade-off of using datasets of smaller sizes to train our model.

{{</citation>}}


### (43/84) Improving Viewpoint Robustness for Visual Recognition via Adversarial Training (Shouwei Ruan et al., 2023)

{{<citation>}}

Shouwei Ruan, Yinpeng Dong, Hang Su, Jianteng Peng, Ning Chen, Xingxing Wei. (2023)  
**Improving Viewpoint Robustness for Visual Recognition via Adversarial Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.11528v1)  

---


**ABSTRACT**  
Viewpoint invariance remains challenging for visual recognition in the 3D world, as altering the viewing directions can significantly impact predictions for the same object. While substantial efforts have been dedicated to making neural networks invariant to 2D image translations and rotations, viewpoint invariance is rarely investigated. Motivated by the success of adversarial training in enhancing model robustness, we propose Viewpoint-Invariant Adversarial Training (VIAT) to improve the viewpoint robustness of image classifiers. Regarding viewpoint transformation as an attack, we formulate VIAT as a minimax optimization problem, where the inner maximization characterizes diverse adversarial viewpoints by learning a Gaussian mixture distribution based on the proposed attack method GMVFool. The outer minimization obtains a viewpoint-invariant classifier by minimizing the expected loss over the worst-case viewpoint distributions that can share the same one for different objects within the same category. Based on GMVFool, we contribute a large-scale dataset called ImageNet-V+ to benchmark viewpoint robustness. Experimental results show that VIAT significantly improves the viewpoint robustness of various image classifiers based on the diversity of adversarial viewpoints generated by GMVFool. Furthermore, we propose ViewRS, a certified viewpoint robustness method that provides a certified radius and accuracy to demonstrate the effectiveness of VIAT from the theoretical perspective.

{{</citation>}}


### (44/84) Redemption from Range-view for Accurate 3D Object Detection (Yihan Wang et al., 2023)

{{<citation>}}

Yihan Wang, Qiao Yan. (2023)  
**Redemption from Range-view for Accurate 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.11482v1)  

---


**ABSTRACT**  
Most recent approaches for 3D object detection predominantly rely on point-view or bird's-eye view representations, with limited exploration of range-view-based methods. The range-view representation suffers from scale variation and surface texture deficiency, both of which pose significant limitations for developing corresponding methods. Notably, the surface texture loss problem has been largely ignored by all existing methods, despite its significant impact on the accuracy of range-view-based 3D object detection. In this study, we propose Redemption from Range-view R-CNN (R2 R-CNN), a novel and accurate approach that comprehensively explores the range-view representation. Our proposed method addresses scale variation through the HD Meta Kernel, which captures range-view geometry information in multiple scales. Additionally, we introduce Feature Points Redemption (FPR) to recover the lost 3D surface texture information from the range view, and Synchronous-Grid RoI Pooling (S-Grid RoI Pooling), a multi-scaled approach with multiple receptive fields for accurate box refinement. Our R2 R-CNN outperforms existing range-view-based methods, achieving state-of-the-art performance on both the KITTI benchmark and the Waymo Open Dataset. Our study highlights the critical importance of addressing the surface texture loss problem for accurate 3D object detection in range-view-based methods. Codes will be made publicly available.

{{</citation>}}


### (45/84) SA-BEV: Generating Semantic-Aware Bird's-Eye-View Feature for Multi-view 3D Object Detection (Jinqing Zhang et al., 2023)

{{<citation>}}

Jinqing Zhang, Yanan Zhang, Qingjie Liu, Yunhong Wang. (2023)  
**SA-BEV: Generating Semantic-Aware Bird's-Eye-View Feature for Multi-view 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.11477v1)  

---


**ABSTRACT**  
Recently, the pure camera-based Bird's-Eye-View (BEV) perception provides a feasible solution for economical autonomous driving. However, the existing BEV-based multi-view 3D detectors generally transform all image features into BEV features, without considering the problem that the large proportion of background information may submerge the object information. In this paper, we propose Semantic-Aware BEV Pooling (SA-BEVPool), which can filter out background information according to the semantic segmentation of image features and transform image features into semantic-aware BEV features. Accordingly, we propose BEV-Paste, an effective data augmentation strategy that closely matches with semantic-aware BEV feature. In addition, we design a Multi-Scale Cross-Task (MSCT) head, which combines task-specific and cross-task information to predict depth distribution and semantic segmentation more accurately, further improving the quality of semantic-aware BEV feature. Finally, we integrate the above modules into a novel multi-view 3D object detection framework, namely SA-BEV. Experiments on nuScenes show that SA-BEV achieves state-of-the-art performance. Code has been available at https://github.com/mengtan00/SA-BEV.git.

{{</citation>}}


### (46/84) Robust Visual Question Answering: Datasets, Methods, and Future Challenges (Jie Ma et al., 2023)

{{<citation>}}

Jie Ma, Pinghui Wang, Dechen Kong, Zewei Wang, Jun Liu, Hongbin Pei, Junzhou Zhao. (2023)  
**Robust Visual Question Answering: Datasets, Methods, and Future Challenges**  

---
Primary Category: cs.CV  
Categories: I-2-10, cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.11471v1)  

---


**ABSTRACT**  
Visual question answering requires a system to provide an accurate natural language answer given an image and a natural language question. However, it is widely recognized that previous generic VQA methods often exhibit a tendency to memorize biases present in the training data rather than learning proper behaviors, such as grounding images before predicting answers. Therefore, these methods usually achieve high in-distribution but poor out-of-distribution performance. In recent years, various datasets and debiasing methods have been proposed to evaluate and enhance the VQA robustness, respectively. This paper provides the first comprehensive survey focused on this emerging fashion. Specifically, we first provide an overview of the development process of datasets from in-distribution and out-of-distribution perspectives. Then, we examine the evaluation metrics employed by these datasets. Thirdly, we propose a typology that presents the development process, similarities and differences, robustness comparison, and technical features of existing debiasing methods. Furthermore, we analyze and discuss the robustness of representative vision-and-language pre-training models on VQA. Finally, through a thorough review of the available literature and experimental analysis, we discuss the key areas for future research from various viewpoints.

{{</citation>}}


### (47/84) Physics-Aware Semi-Supervised Underwater Image Enhancement (Hao Qi et al., 2023)

{{<citation>}}

Hao Qi, Xinghui Dong. (2023)  
**Physics-Aware Semi-Supervised Underwater Image Enhancement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.11470v1)  

---


**ABSTRACT**  
Underwater images normally suffer from degradation due to the transmission medium of water bodies. Both traditional prior-based approaches and deep learning-based methods have been used to address this problem. However, the inflexible assumption of the former often impairs their effectiveness in handling diverse underwater scenes, while the generalization of the latter to unseen images is usually weakened by insufficient data. In this study, we leverage both the physics-based underwater Image Formation Model (IFM) and deep learning techniques for Underwater Image Enhancement (UIE). To this end, we propose a novel Physics-Aware Dual-Stream Underwater Image Enhancement Network, i.e., PA-UIENet, which comprises a Transmission Estimation Steam (T-Stream) and an Ambient Light Estimation Stream (A-Stream). This network fulfills the UIE task by explicitly estimating the degradation parameters of the IFM. We also adopt an IFM-inspired semi-supervised learning framework, which exploits both the labeled and unlabeled images, to address the issue of insufficient data. Our method performs better than, or at least comparably to, eight baselines across five testing sets in the degradation estimation and UIE tasks. This should be due to the fact that it not only can model the degradation but also can learn the characteristics of diverse underwater scenes.

{{</citation>}}


### (48/84) Distribution Shift Matters for Knowledge Distillation with Webly Collected Images (Jialiang Tang et al., 2023)

{{<citation>}}

Jialiang Tang, Shuo Chen, Gang Niu, Masashi Sugiyama, Chen Gong. (2023)  
**Distribution Shift Matters for Knowledge Distillation with Webly Collected Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.11469v1)  

---


**ABSTRACT**  
Knowledge distillation aims to learn a lightweight student network from a pre-trained teacher network. In practice, existing knowledge distillation methods are usually infeasible when the original training data is unavailable due to some privacy issues and data management considerations. Therefore, data-free knowledge distillation approaches proposed to collect training instances from the Internet. However, most of them have ignored the common distribution shift between the instances from original training data and webly collected data, affecting the reliability of the trained student network. To solve this problem, we propose a novel method dubbed ``Knowledge Distillation between Different Distributions" (KD$^{3}$), which consists of three components. Specifically, we first dynamically select useful training instances from the webly collected data according to the combined predictions of teacher network and student network. Subsequently, we align both the weighted features and classifier parameters of the two networks for knowledge memorization. Meanwhile, we also build a new contrastive learning block called MixDistribution to generate perturbed data with a new distribution for instance alignment, so that the student network can further learn a distribution-invariant representation. Intensive experiments on various benchmark datasets demonstrate that our proposed KD$^{3}$ can outperform the state-of-the-art data-free knowledge distillation approaches.

{{</citation>}}


### (49/84) Strip-MLP: Efficient Token Interaction for Vision MLP (Guiping Cao et al., 2023)

{{<citation>}}

Guiping Cao, Shengda Luo, Wenjian Huang, Xiangyuan Lan, Dongmei Jiang, Yaowei Wang, Jianguo Zhang. (2023)  
**Strip-MLP: Efficient Token Interaction for Vision MLP**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.11458v1)  

---


**ABSTRACT**  
Token interaction operation is one of the core modules in MLP-based models to exchange and aggregate information between different spatial locations. However, the power of token interaction on the spatial dimension is highly dependent on the spatial resolution of the feature maps, which limits the model's expressive ability, especially in deep layers where the feature are down-sampled to a small spatial size. To address this issue, we present a novel method called \textbf{Strip-MLP} to enrich the token interaction power in three ways. Firstly, we introduce a new MLP paradigm called Strip MLP layer that allows the token to interact with other tokens in a cross-strip manner, enabling the tokens in a row (or column) to contribute to the information aggregations in adjacent but different strips of rows (or columns). Secondly, a \textbf{C}ascade \textbf{G}roup \textbf{S}trip \textbf{M}ixing \textbf{M}odule (CGSMM) is proposed to overcome the performance degradation caused by small spatial feature size. The module allows tokens to interact more effectively in the manners of within-patch and cross-patch, which is independent to the feature spatial size. Finally, based on the Strip MLP layer, we propose a novel \textbf{L}ocal \textbf{S}trip \textbf{M}ixing \textbf{M}odule (LSMM) to boost the token interaction power in the local region. Extensive experiments demonstrate that Strip-MLP significantly improves the performance of MLP-based models on small datasets and obtains comparable or even better results on ImageNet. In particular, Strip-MLP models achieve higher average Top-1 accuracy than existing MLP-based models by +2.44\% on Caltech-101 and +2.16\% on CIFAR-100. The source codes will be available at~\href{https://github.com/Med-Process/Strip_MLP{https://github.com/Med-Process/Strip\_MLP}.

{{</citation>}}


### (50/84) Attention Consistency Refined Masked Frequency Forgery Representation for Generalizing Face Forgery Detection (Decheng Liu et al., 2023)

{{<citation>}}

Decheng Liu, Tao Chen, Chunlei Peng, Nannan Wang, Ruimin Hu, Xinbo Gao. (2023)  
**Attention Consistency Refined Masked Frequency Forgery Representation for Generalizing Face Forgery Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.11438v1)  

---


**ABSTRACT**  
Due to the successful development of deep image generation technology, visual data forgery detection would play a more important role in social and economic security. Existing forgery detection methods suffer from unsatisfactory generalization ability to determine the authenticity in the unseen domain. In this paper, we propose a novel Attention Consistency Refined masked frequency forgery representation model toward generalizing face forgery detection algorithm (ACMF). Most forgery technologies always bring in high-frequency aware cues, which make it easy to distinguish source authenticity but difficult to generalize to unseen artifact types. The masked frequency forgery representation module is designed to explore robust forgery cues by randomly discarding high-frequency information. In addition, we find that the forgery attention map inconsistency through the detection network could affect the generalizability. Thus, the forgery attention consistency is introduced to force detectors to focus on similar attention regions for better generalization ability. Experiment results on several public face forgery datasets (FaceForensic++, DFD, Celeb-DF, and WDF datasets) demonstrate the superior performance of the proposed method compared with the state-of-the-art methods.

{{</citation>}}


### (51/84) Deep Directly-Trained Spiking Neural Networks for Object Detection (Qiaoyi Su et al., 2023)

{{<citation>}}

Qiaoyi Su, Yuhong Chou, Yifan Hu, Jianing Li, Shijie Mei, Ziyang Zhang, Guoqi Li. (2023)  
**Deep Directly-Trained Spiking Neural Networks for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.11411v2)  

---


**ABSTRACT**  
Spiking neural networks (SNNs) are brain-inspired energy-efficient models that encode information in spatiotemporal dynamics. Recently, deep SNNs trained directly have shown great success in achieving high performance on classification tasks with very few time steps. However, how to design a directly-trained SNN for the regression task of object detection still remains a challenging problem. To address this problem, we propose EMS-YOLO, a novel directly-trained SNN framework for object detection, which is the first trial to train a deep SNN with surrogate gradients for object detection rather than ANN-SNN conversion strategies. Specifically, we design a full-spike residual block, EMS-ResNet, which can effectively extend the depth of the directly-trained SNN with low power consumption. Furthermore, we theoretically analyze and prove the EMS-ResNet could avoid gradient vanishing or exploding. The results demonstrate that our approach outperforms the state-of-the-art ANN-SNN conversion methods (at least 500 time steps) in extremely fewer time steps (only 4 time steps). It is shown that our model could achieve comparable performance to the ANN with the same architecture while consuming 5.83 times less energy on the frame-based COCO Dataset and the event-based Gen1 Dataset.

{{</citation>}}


### (52/84) Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning (Jian Ma et al., 2023)

{{<citation>}}

Jian Ma, Junhao Liang, Chen Chen, Haonan Lu. (2023)  
**Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11410v1)  

---


**ABSTRACT**  
Recent progress in personalized image generation using diffusion models has been significant. However, development in the area of open-domain and non-fine-tuning personalized image generation is proceeding rather slowly. In this paper, we propose Subject-Diffusion, a novel open-domain personalized image generation model that, in addition to not requiring test-time fine-tuning, also only requires a single reference image to support personalized generation of single- or multi-subject in any domain. Firstly, we construct an automatic data labeling tool and use the LAION-Aesthetics dataset to construct a large-scale dataset consisting of 76M images and their corresponding subject detection bounding boxes, segmentation masks and text descriptions. Secondly, we design a new unified framework that combines text and image semantics by incorporating coarse location and fine-grained reference image control to maximize subject fidelity and generalization. Furthermore, we also adopt an attention control mechanism to support multi-subject generation. Extensive qualitative and quantitative results demonstrate that our method outperforms other SOTA frameworks in single, multiple, and human customized image generation. Please refer to our \href{https://oppo-mente-lab.github.io/subject_diffusion/}{project page}

{{</citation>}}


### (53/84) LatentAugment: Data Augmentation via Guided Manipulation of GAN's Latent Space (Lorenzo Tronchin et al., 2023)

{{<citation>}}

Lorenzo Tronchin, Minh H. Vu, Paolo Soda, Tommy Löfstedt. (2023)  
**LatentAugment: Data Augmentation via Guided Manipulation of GAN's Latent Space**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.11375v1)  

---


**ABSTRACT**  
Data Augmentation (DA) is a technique to increase the quantity and diversity of the training data, and by that alleviate overfitting and improve generalisation. However, standard DA produces synthetic data for augmentation with limited diversity. Generative Adversarial Networks (GANs) may unlock additional information in a dataset by generating synthetic samples having the appearance of real images. However, these models struggle to simultaneously address three key requirements: fidelity and high-quality samples; diversity and mode coverage; and fast sampling. Indeed, GANs generate high-quality samples rapidly, but have poor mode coverage, limiting their adoption in DA applications. We propose LatentAugment, a DA strategy that overcomes the low diversity of GANs, opening up for use in DA applications. Without external supervision, LatentAugment modifies latent vectors and moves them into latent space regions to maximise the synthetic images' diversity and fidelity. It is also agnostic to the dataset and the downstream task. A wide set of experiments shows that LatentAugment improves the generalisation of a deep model translating from MRI-to-CT beating both standard DA as well GAN-based sampling. Moreover, still in comparison with GAN-based sampling, LatentAugment synthetic samples show superior mode coverage and diversity. Code is available at: https://github.com/ltronchin/LatentAugment.

{{</citation>}}


### (54/84) ParGANDA: Making Synthetic Pedestrians A Reality For Object Detection (Daria Reshetova et al., 2023)

{{<citation>}}

Daria Reshetova, Guanhang Wu, Marcel Puyat, Chunhui Gu, Huizhong Chen. (2023)  
**ParGANDA: Making Synthetic Pedestrians A Reality For Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.11360v1)  

---


**ABSTRACT**  
Object detection is the key technique to a number of Computer Vision applications, but it often requires large amounts of annotated data to achieve decent results. Moreover, for pedestrian detection specifically, the collected data might contain some personally identifiable information (PII), which is highly restricted in many countries. This label intensive and privacy concerning task has recently led to an increasing interest in training the detection models using synthetically generated pedestrian datasets collected with a photo-realistic video game engine. The engine is able to generate unlimited amounts of data with precise and consistent annotations, which gives potential for significant gains in the real-world applications. However, the use of synthetic data for training introduces a synthetic-to-real domain shift aggravating the final performance. To close the gap between the real and synthetic data, we propose to use a Generative Adversarial Network (GAN), which performsparameterized unpaired image-to-image translation to generate more realistic images. The key benefit of using the GAN is its intrinsic preference of low-level changes to geometric ones, which means annotations of a given synthetic image remain accurate even after domain translation is performed thus eliminating the need for labeling real data. We extensively experimented with the proposed method using MOTSynth dataset to train and MOT17 and MOT20 detection datasets to test, with experimental results demonstrating the effectiveness of this method. Our approach not only produces visually plausible samples but also does not require any labels of the real domain thus making it applicable to the variety of downstream tasks.

{{</citation>}}


### (55/84) Generating Image-Specific Text Improves Fine-grained Image Classification (Emily Mu et al., 2023)

{{<citation>}}

Emily Mu, Kathleen M. Lewis, Adrian V. Dalca, John Guttag. (2023)  
**Generating Image-Specific Text Improves Fine-grained Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.11315v1)  

---


**ABSTRACT**  
Recent vision-language models outperform vision-only models on many image classification tasks. However, because of the absence of paired text/image descriptions, it remains difficult to fine-tune these models for fine-grained image classification. In this work, we propose a method, GIST, for generating image-specific fine-grained text descriptions from image-only datasets, and show that these text descriptions can be used to improve classification. Key parts of our method include 1. prompting a pretrained large language model with domain-specific prompts to generate diverse fine-grained text descriptions for each class and 2. using a pretrained vision-language model to match each image to label-preserving text descriptions that capture relevant visual features in the image. We demonstrate the utility of GIST by fine-tuning vision-language models on the image-and-generated-text pairs to learn an aligned vision-language representation space for improved classification. We evaluate our learned representation space in full-shot and few-shot scenarios across four diverse fine-grained classification datasets, each from a different domain. Our method achieves an average improvement of $4.1\%$ in accuracy over CLIP linear probes and an average of $1.1\%$ improvement in accuracy over the previous state-of-the-art image-text classification method on the full-shot datasets. Our method achieves similar improvements across few-shot regimes. Code is available at https://github.com/emu1729/GIST.

{{</citation>}}


## cs.CR (1)



### (56/84) Exploring Security Commits in Python (Shiyu Sun et al., 2023)

{{<citation>}}

Shiyu Sun, Shu Wang, Xinda Wang, Yunlong Xing, Elisa Zhang, Kun Sun. (2023)  
**Exploring Security Commits in Python**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.11853v1)  

---


**ABSTRACT**  
Python has become the most popular programming language as it is friendly to work with for beginners. However, a recent study has found that most security issues in Python have not been indexed by CVE and may only be fixed by 'silent' security commits, which pose a threat to software security and hinder the security fixes to downstream software. It is critical to identify the hidden security commits; however, the existing datasets and methods are insufficient for security commit detection in Python, due to the limited data variety, non-comprehensive code semantics, and uninterpretable learned features. In this paper, we construct the first security commit dataset in Python, namely PySecDB, which consists of three subsets including a base dataset, a pilot dataset, and an augmented dataset. The base dataset contains the security commits associated with CVE records provided by MITRE. To increase the variety of security commits, we build the pilot dataset from GitHub by filtering keywords within the commit messages. Since not all commits provide commit messages, we further construct the augmented dataset by understanding the semantics of code changes. To build the augmented dataset, we propose a new graph representation named CommitCPG and a multi-attributed graph learning model named SCOPY to identify the security commit candidates through both sequential and structural code semantics. The evaluation shows our proposed algorithms can improve the data collection efficiency by up to 40 percentage points. After manual verification by three security experts, PySecDB consists of 1,258 security commits and 2,791 non-security commits. Furthermore, we conduct an extensive case study on PySecDB and discover four common security fix patterns that cover over 85% of security commits in Python, providing insight into secure software maintenance, vulnerability detection, and automated program repair.

{{</citation>}}


## cs.CL (9)



### (57/84) MythQA: Query-Based Large-Scale Check-Worthy Claim Detection through Multi-Answer Open-Domain Question Answering (Yang Bai et al., 2023)

{{<citation>}}

Yang Bai, Anthony Colas, Daisy Zhe Wang. (2023)  
**MythQA: Query-Based Large-Scale Check-Worthy Claim Detection through Multi-Answer Open-Domain Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: NLP, QA, Question Answering, Twitter  
[Paper Link](http://arxiv.org/abs/2307.11848v1)  

---


**ABSTRACT**  
Check-worthy claim detection aims at providing plausible misinformation to downstream fact-checking systems or human experts to check. This is a crucial step toward accelerating the fact-checking process. Many efforts have been put into how to identify check-worthy claims from a small scale of pre-collected claims, but how to efficiently detect check-worthy claims directly from a large-scale information source, such as Twitter, remains underexplored. To fill this gap, we introduce MythQA, a new multi-answer open-domain question answering(QA) task that involves contradictory stance mining for query-based large-scale check-worthy claim detection. The idea behind this is that contradictory claims are a strong indicator of misinformation that merits scrutiny by the appropriate authorities. To study this task, we construct TweetMythQA, an evaluation dataset containing 522 factoid multi-answer questions based on controversial topics. Each question is annotated with multiple answers. Moreover, we collect relevant tweets for each distinct answer, then classify them into three categories: "Supporting", "Refuting", and "Neutral". In total, we annotated 5.3K tweets. Contradictory evidence is collected for all answers in the dataset. Finally, we present a baseline system for MythQA and evaluate existing NLP models for each system component using the TweetMythQA dataset. We provide initial benchmarks and identify key challenges for future models to improve upon. Code and data are available at: https://github.com/TonyBY/Myth-QA

{{</citation>}}


### (58/84) Multimodal Document Analytics for Banking Process Automation (Christopher Gerling et al., 2023)

{{<citation>}}

Christopher Gerling, Stefan Lessmann. (2023)  
**Multimodal Document Analytics for Banking Process Automation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL, q-fin-CP  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.11845v1)  

---


**ABSTRACT**  
In response to growing FinTech competition and the need for improved operational efficiency, this research focuses on understanding the potential of advanced document analytics, particularly using multimodal models, in banking processes. We perform a comprehensive analysis of the diverse banking document landscape, highlighting the opportunities for efficiency gains through automation and advanced analytics techniques in the customer business. Building on the rapidly evolving field of natural language processing (NLP), we illustrate the potential of models such as LayoutXLM, a cross-lingual, multimodal, pre-trained model, for analyzing diverse documents in the banking sector. This model performs a text token classification on German company register extracts with an overall F1 score performance of around 80\%. Our empirical evidence confirms the critical role of layout information in improving model performance and further underscores the benefits of integrating image information. Interestingly, our study shows that over 75% F1 score can be achieved with only 30% of the training data, demonstrating the efficiency of LayoutXLM. Through addressing state-of-the-art document analysis frameworks, our study aims to enhance process efficiency and demonstrate the real-world applicability and benefits of multimodal models within banking.

{{</citation>}}


### (59/84) OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples (Ryuto Koike et al., 2023)

{{<citation>}}

Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki. (2023)  
**OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.11729v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, the effectiveness of these detectors in real-life situations, such as when students use LLMs for writing homework assignments (e.g., essays) and quickly learn how to evade these detectors, has not been explored. In this paper, we propose OUTFOX, a novel framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output and apply this to the domain of student essays. In our framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect. While the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Our experiments show that our proposed detector learned in-context from the attacker improves the detection performance on the attacked dataset by up to +41.3 point F1-score. While our proposed attacker can drastically degrade the performance of the detector by up to -57.0 point F1-score compared to the paraphrasing method.

{{</citation>}}


### (60/84) CausE: Towards Causal Knowledge Graph Embedding (Yichi Zhang et al., 2023)

{{<citation>}}

Yichi Zhang, Wen Zhang. (2023)  
**CausE: Towards Causal Knowledge Graph Embedding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2307.11610v2)  

---


**ABSTRACT**  
Knowledge graph embedding (KGE) focuses on representing the entities and relations of a knowledge graph (KG) into the continuous vector spaces, which can be employed to predict the missing triples to achieve knowledge graph completion (KGC). However, KGE models often only briefly learn structural correlations of triple data and embeddings would be misled by the trivial patterns and noisy links in real-world KGs. To address this issue, we build the new paradigm of KGE in the context of causality and embedding disentanglement. We further propose a Causality-enhanced knowledge graph Embedding (CausE) framework. CausE employs causal intervention to estimate the causal effect of the confounder embeddings and design new training objectives to make stable predictions. Experimental results demonstrate that CausE could outperform the baseline models and achieve state-of-the-art KGC performance. We release our code in https://github.com/zjukg/CausE.

{{</citation>}}


### (61/84) Incorporating Human Translator Style into English-Turkish Literary Machine Translation (Zeynep Yirmibeşoğlu et al., 2023)

{{<citation>}}

Zeynep Yirmibeşoğlu, Olgun Dursun, Harun Dallı, Mehmet Şahin, Ena Hodzik, Sabri Gürses, Tunga Güngör. (2023)  
**Incorporating Human Translator Style into English-Turkish Literary Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2307.11457v1)  

---


**ABSTRACT**  
Although machine translation systems are mostly designed to serve in the general domain, there is a growing tendency to adapt these systems to other domains like literary translation. In this paper, we focus on English-Turkish literary translation and develop machine translation models that take into account the stylistic features of translators. We fine-tune a pre-trained machine translation model by the manually-aligned works of a particular translator. We make a detailed analysis of the effects of manual and automatic alignments, data augmentation methods, and corpus size on the translations. We propose an approach based on stylistic features to evaluate the style of a translator in the output translations. We show that the human translator style can be highly recreated in the target machine translations by adapting the models to the style of the translator.

{{</citation>}}


### (62/84) Is ChatGPT Involved in Texts? Measure the Polish Ratio to Detect ChatGPT-Generated Text (Lingyi Yang et al., 2023)

{{<citation>}}

Lingyi Yang, Feng Jiang, Haizhou Li. (2023)  
**Is ChatGPT Involved in Texts? Measure the Polish Ratio to Detect ChatGPT-Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.11380v1)  

---


**ABSTRACT**  
The remarkable capabilities of large-scale language models, such as ChatGPT, in text generation have incited awe and spurred researchers to devise detectors to mitigate potential risks, including misinformation, phishing, and academic dishonesty. Despite this, most previous studies, including HC3, have been predominantly geared towards creating detectors that differentiate between purely ChatGPT-generated texts and human-authored texts. This approach, however, fails to work on discerning texts generated through human-machine collaboration, such as ChatGPT-polished texts. Addressing this gap, we introduce a novel dataset termed HPPT (ChatGPT-polished academic abstracts), facilitating the construction of more robust detectors. It diverges from extant corpora by comprising pairs of human-written and ChatGPT-polished abstracts instead of purely ChatGPT-generated texts. Additionally, we propose the "Polish Ratio" method, an innovative measure of ChatGPT's involvement in text generation based on editing distance. It provides a mechanism to measure the degree of human originality in the resulting text. Our experimental results show our proposed model has better robustness on the HPPT dataset and two existing datasets (HC3 and CDB). Furthermore, the "Polish Ratio" we proposed offers a more comprehensive explanation by quantifying the degree of ChatGPT involvement, which indicates that a Polish Ratio value greater than 0.2 signifies ChatGPT involvement and a value exceeding 0.6 implies that ChatGPT generates most of the text.

{{</citation>}}


### (63/84) CohortGPT: An Enhanced GPT for Participant Recruitment in Clinical Study (Zihan Guan et al., 2023)

{{<citation>}}

Zihan Guan, Zihao Wu, Zhengliang Liu, Dufan Wu, Hui Ren, Quanzheng Li, Xiang Li, Ninghao Liu. (2023)  
**CohortGPT: An Enhanced GPT for Participant Recruitment in Clinical Study**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, Clinical, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.11346v1)  

---


**ABSTRACT**  
Participant recruitment based on unstructured medical texts such as clinical notes and radiology reports has been a challenging yet important task for the cohort establishment in clinical research. Recently, Large Language Models (LLMs) such as ChatGPT have achieved tremendous success in various downstream tasks thanks to their promising performance in language understanding, inference, and generation. It is then natural to test their feasibility in solving the cohort recruitment task, which involves the classification of a given paragraph of medical text into disease label(s). However, when applied to knowledge-intensive problem settings such as medical text classification, where the LLMs are expected to understand the decision made by human experts and accurately identify the implied disease labels, the LLMs show a mediocre performance. A possible explanation is that, by only using the medical text, the LLMs neglect to use the rich context of additional information that languages afford. To this end, we propose to use a knowledge graph as auxiliary information to guide the LLMs in making predictions. Moreover, to further boost the LLMs adapt to the problem setting, we apply a chain-of-thought (CoT) sample selection strategy enhanced by reinforcement learning, which selects a set of CoT samples given each individual medical report. Experimental results and various ablation studies show that our few-shot learning method achieves satisfactory performance compared with fine-tuning strategies and gains superb advantages when the available data is limited. The code and sample dataset of the proposed CohortGPT model is available at: https://anonymous.4open.science/r/CohortGPT-4872/

{{</citation>}}


### (64/84) Making Pre-trained Language Models both Task-solvers and Self-calibrators (Yangyi Chen et al., 2023)

{{<citation>}}

Yangyi Chen, Xingyao Wang, Heng Ji. (2023)  
**Making Pre-trained Language Models both Task-solvers and Self-calibrators**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.11316v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) serve as backbones for various real-world systems. For high-stake applications, it's equally essential to have reasonable confidence estimations in predictions. While the vanilla confidence scores of PLMs can already be effectively utilized, PLMs consistently become overconfident in their wrong predictions, which is not desirable in practice. Previous work shows that introducing an extra calibration task can mitigate this issue. The basic idea involves acquiring additional data to train models in predicting the confidence of their initial predictions. However, it only demonstrates the feasibility of this kind of method, assuming that there are abundant extra available samples for the introduced calibration task. In this work, we consider the practical scenario that we need to effectively utilize training samples to make PLMs both task-solvers and self-calibrators. Three challenges are presented, including limited training samples, data imbalance, and distribution shifts. We first conduct pilot experiments to quantify various decisive factors in the calibration task. Based on the empirical analysis results, we propose a training algorithm LM-TOAST to tackle the challenges. Experimental results show that LM-TOAST can effectively utilize the training data to make PLMs have reasonable confidence estimations while maintaining the original task performance. Further, we consider three downstream applications, namely selective classification, adversarial defense, and model cascading, to show the practical usefulness of LM-TOAST. The code will be made public at \url{https://github.com/Yangyi-Chen/LM-TOAST}.

{{</citation>}}


### (65/84) Generator-Retriever-Generator: A Novel Approach to Open-domain Question Answering (Abdelrahman Abdallah et al., 2023)

{{<citation>}}

Abdelrahman Abdallah, Adam Jatowt. (2023)  
**Generator-Retriever-Generator: A Novel Approach to Open-domain Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.11278v1)  

---


**ABSTRACT**  
Open-domain question answering (QA) tasks usually require the retrieval of relevant information from a large corpus to generate accurate answers. We propose a novel approach called Generator-Retriever-Generator (GRG) that combines document retrieval techniques with a large language model (LLM), by first prompting the model to generate contextual documents based on a given question. In parallel, a dual-encoder network retrieves documents that are relevant to the question from an external corpus. The generated and retrieved documents are then passed to the second LLM, which generates the final answer. By combining document retrieval and LLM generation, our approach addresses the challenges of open-domain QA, such as generating informative and contextually relevant answers. GRG outperforms the state-of-the-art generate-then-read and retrieve-then-read pipelines (GENREAD and RFiD) improving their performance at least by +5.2, +4.2, and +1.6 on TriviaQA, NQ, and WebQ datasets, respectively. We provide code, datasets, and checkpoints \footnote{\url{https://github.com/abdoelsayed2016/GRG}}

{{</citation>}}


## cs.AI (9)



### (66/84) eXplainable Artificial Intelligence (XAI) in age prediction: A systematic review (Alena Kalyakulina et al., 2023)

{{<citation>}}

Alena Kalyakulina, Igor Yusipov. (2023)  
**eXplainable Artificial Intelligence (XAI) in age prediction: A systematic review**  

---
Primary Category: cs.AI  
Categories: I-2-1; J-3, cs-AI, cs-LG, cs.AI, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13704v1)  

---


**ABSTRACT**  
eXplainable Artificial Intelligence (XAI) is now an important and essential part of machine learning, allowing to explain the predictions of complex models. XAI is especially required in risky applications, particularly in health care, where human lives depend on the decisions of AI systems. One area of medical research is age prediction and identification of biomarkers of aging and age-related diseases. However, the role of XAI in the age prediction task has not previously been explored directly. In this review, we discuss the application of XAI approaches to age prediction tasks. We give a systematic review of the works organized by body systems, and discuss the benefits of XAI in medical applications and, in particular, in the age prediction domain.

{{</citation>}}


### (67/84) Statement-based Memory for Neural Source Code Summarization (Aakash Bansal et al., 2023)

{{<citation>}}

Aakash Bansal, Siyuan Jiang, Sakib Haque, Collin McMillan. (2023)  
**Statement-based Memory for Neural Source Code Summarization**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Summarization, Transformer  
[Paper Link](http://arxiv.org/abs/2307.11709v1)  

---


**ABSTRACT**  
Source code summarization is the task of writing natural language descriptions of source code behavior. Code summarization underpins software documentation for programmers. Short descriptions of code help programmers understand the program quickly without having to read the code itself. Lately, neural source code summarization has emerged as the frontier of research into automated code summarization techniques. By far the most popular targets for summarization are program subroutines. The idea, in a nutshell, is to train an encoder-decoder neural architecture using large sets of examples of subroutines extracted from code repositories. The encoder represents the code and the decoder represents the summary. However, most current approaches attempt to treat the subroutine as a single unit. For example, by taking the entire subroutine as input to a Transformer or RNN-based encoder. But code behavior tends to depend on the flow from statement to statement. Normally dynamic analysis may shed light on this flow, but dynamic analysis on hundreds of thousands of examples in large datasets is not practical. In this paper, we present a statement-based memory encoder that learns the important elements of flow during training, leading to a statement-based subroutine representation without the need for dynamic analysis. We implement our encoder for code summarization and demonstrate a significant improvement over the state-of-the-art.

{{</citation>}}


### (68/84) Identifying Relevant Features of CSE-CIC-IDS2018 Dataset for the Development of an Intrusion Detection System (László Göcs et al., 2023)

{{<citation>}}

László Göcs, Zsolt Csaba Johanyák. (2023)  
**Identifying Relevant Features of CSE-CIC-IDS2018 Dataset for the Development of an Intrusion Detection System**  

---
Primary Category: cs.AI  
Categories: 68T05, I-2, cs-AI, cs.AI  
Keywords: AWS, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2307.11544v1)  

---


**ABSTRACT**  
Intrusion detection systems (IDSs) are essential elements of IT systems. Their key component is a classification module that continuously evaluates some features of the network traffic and identifies possible threats. Its efficiency is greatly affected by the right selection of the features to be monitored. Therefore, the identification of a minimal set of features that are necessary to safely distinguish malicious traffic from benign traffic is indispensable in the course of the development of an IDS. This paper presents the preprocessing and feature selection workflow as well as its results in the case of the CSE-CIC-IDS2018 on AWS dataset, focusing on five attack types. To identify the relevant features, six feature selection methods were applied, and the final ranking of the features was elaborated based on their average score. Next, several subsets of the features were formed based on different ranking threshold values, and each subset was tried with five classification algorithms to determine the optimal feature set for each attack type. During the evaluation, four widely used metrics were taken into consideration.

{{</citation>}}


### (69/84) Model Reporting for Certifiable AI: A Proposal from Merging EU Regulation into AI Development (Danilo Brajovic et al., 2023)

{{<citation>}}

Danilo Brajovic, Niclas Renner, Vincent Philipp Goebels, Philipp Wagner, Benjamin Fresz, Martin Biller, Mara Klaeb, Janika Kutz, Jens Neuhuettler, Marco F. Huber. (2023)  
**Model Reporting for Certifiable AI: A Proposal from Merging EU Regulation into AI Development**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11525v1)  

---


**ABSTRACT**  
Despite large progress in Explainable and Safe AI, practitioners suffer from a lack of regulation and standards for AI safety. In this work we merge recent regulation efforts by the European Union and first proposals for AI guidelines with recent trends in research: data and model cards. We propose the use of standardized cards to document AI applications throughout the development process. Our main contribution is the introduction of use-case and operation cards, along with updates for data and model cards to cope with regulatory requirements. We reference both recent research as well as the source of the regulation in our cards and provide references to additional support material and toolboxes whenever possible. The goal is to design cards that help practitioners develop safe AI systems throughout the development process, while enabling efficient third-party auditing of AI applications, being easy to understand, and building trust in the system. Our work incorporates insights from interviews with certification experts as well as developers and individuals working with the developed AI applications.

{{</citation>}}


### (70/84) IndigoVX: Where Human Intelligence Meets AI for Optimal Decision Making (Kais Dukes, 2023)

{{<citation>}}

Kais Dukes. (2023)  
**IndigoVX: Where Human Intelligence Meets AI for Optimal Decision Making**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11516v1)  

---


**ABSTRACT**  
This paper defines a new approach for augmenting human intelligence with AI for optimal goal solving. Our proposed AI, Indigo, is an acronym for Informed Numerical Decision-making through Iterative Goal-Oriented optimization. When combined with a human collaborator, we term the joint system IndigoVX, for Virtual eXpert. The system is conceptually simple. We envisage this method being applied to games or business strategies, with the human providing strategic context and the AI offering optimal, data-driven moves. Indigo operates through an iterative feedback loop, harnessing the human expert's contextual knowledge and the AI's data-driven insights to craft and refine strategies towards a well-defined goal. Using a quantified three-score schema, this hybridization allows the combined team to evaluate strategies and refine their plan, while adapting to challenges and changes in real-time.

{{</citation>}}


### (71/84) Zero-touch realization of Pervasive Artificial Intelligence-as-a-service in 6G networks (Emna Baccour et al., 2023)

{{<citation>}}

Emna Baccour, Mhd Saria Allahham, Aiman Erbad, Amr Mohamed, Ahmed Refaey Hussein, Mounir Hamdi. (2023)  
**Zero-touch realization of Pervasive Artificial Intelligence-as-a-service in 6G networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DC, cs-NI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11468v1)  

---


**ABSTRACT**  
The vision of the upcoming 6G technologies, characterized by ultra-dense network, low latency, and fast data rate is to support Pervasive AI (PAI) using zero-touch solutions enabling self-X (e.g., self-configuration, self-monitoring, and self-healing) services. However, the research on 6G is still in its infancy, and only the first steps have been taken to conceptualize its design, investigate its implementation, and plan for use cases. Toward this end, academia and industry communities have gradually shifted from theoretical studies of AI distribution to real-world deployment and standardization. Still, designing an end-to-end framework that systematizes the AI distribution by allowing easier access to the service using a third-party application assisted by a zero-touch service provisioning has not been well explored. In this context, we introduce a novel platform architecture to deploy a zero-touch PAI-as-a-Service (PAIaaS) in 6G networks supported by a blockchain-based smart system. This platform aims to standardize the pervasive AI at all levels of the architecture and unify the interfaces in order to facilitate the service deployment across application and infrastructure domains, relieve the users worries about cost, security, and resource allocation, and at the same time, respect the 6G stringent performance requirements. As a proof of concept, we present a Federated Learning-as-a-service use case where we evaluate the ability of our proposed system to self-optimize and self-adapt to the dynamics of 6G networks in addition to minimizing the users' perceived costs.

{{</citation>}}


### (72/84) AIGC Empowering Telecom Sector White Paper_chinese (Ye Ouyang et al., 2023)

{{<citation>}}

Ye Ouyang, Yaqin Zhang, Xiaozhou Ye, Yunxin Liu, Yong Song, Yang Liu, Sen Bian, Zhiyong Liu. (2023)  
**AIGC Empowering Telecom Sector White Paper_chinese**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2307.11449v2)  

---


**ABSTRACT**  
In the global craze of GPT, people have deeply realized that AI, as a transformative technology and key force in economic and social development, will bring great leaps and breakthroughs to the global industry and profoundly influence the future world competition pattern. As the builder and operator of information and communication infrastructure, the telecom sector provides infrastructure support for the development of AI, and even takes the lead in the implementation of AI applications. How to enable the application of AIGC (GPT) and implement AIGC in the telecom sector are questions that telecom practitioners must ponder and answer. Through the study of GPT, a typical representative of AIGC, the authors have analyzed how GPT empowers the telecom sector in the form of scenarios, discussed the gap between the current GPT general model and telecom services, proposed for the first time a Telco Augmented Cognition capability system, provided answers to how to construct a telecom service GPT in the telecom sector, and carried out various practices. Our counterparts in the industry are expected to focus on collaborative innovation around telecom and AI, build an open and shared innovation ecosystem, promote the deep integration of AI and telecom sector, and accelerate the construction of next-generation information infrastructure, in an effort to facilitate the digital transformation of the economy and society.

{{</citation>}}


### (73/84) A Two-stage Fine-tuning Strategy for Generalizable Manipulation Skill of Embodied AI (Fang Gao et al., 2023)

{{<citation>}}

Fang Gao, XueTao Li, Jun Yu, Feng Shaung. (2023)  
**A Two-stage Fine-tuning Strategy for Generalizable Manipulation Skill of Embodied AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2307.11343v1)  

---


**ABSTRACT**  
The advent of Chat-GPT has led to a surge of interest in Embodied AI. However, many existing Embodied AI models heavily rely on massive interactions with training environments, which may not be practical in real-world situations. To this end, the Maniskill2 has introduced a full-physics simulation benchmark for manipulating various 3D objects. This benchmark enables agents to be trained using diverse datasets of demonstrations and evaluates their ability to generalize to unseen scenarios in testing environments. In this paper, we propose a novel two-stage fine-tuning strategy that aims to further enhance the generalization capability of our model based on the Maniskill2 benchmark. Through extensive experiments, we demonstrate the effectiveness of our approach by achieving the 1st prize in all three tracks of the ManiSkill2 Challenge. Our findings highlight the potential of our method to improve the generalization abilities of Embodied AI models and pave the way for their ractical applications in real-world scenarios. All codes and models of our solution is available at https://github.com/xtli12/GXU-LIPE.git

{{</citation>}}


### (74/84) Eliminating Unintended Stable Fixpoints for Hybrid Reasoning Systems (Spencer Killen et al., 2023)

{{<citation>}}

Spencer Killen, Jia-Huai You. (2023)  
**Eliminating Unintended Stable Fixpoints for Hybrid Reasoning Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.11286v1)  

---


**ABSTRACT**  
A wide variety of nonmonotonic semantics can be expressed as approximators defined under AFT (Approximation Fixpoint Theory). Using traditional AFT theory, it is not possible to define approximators that rely on information computed in previous iterations of stable revision. However, this information is rich for semantics that incorporate classical negation into nonmonotonic reasoning. In this work, we introduce a methodology resembling AFT that can utilize priorly computed upper bounds to more precisely capture semantics. We demonstrate our framework's applicability to hybrid MKNF (minimal knowledge and negation as failure) knowledge bases by extending the state-of-the-art approximator.

{{</citation>}}


## cs.CE (1)



### (75/84) PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks (Leo Zhiyuan Zhao et al., 2023)

{{<citation>}}

Leo Zhiyuan Zhao, Xueying Ding, B. Aditya Prakash. (2023)  
**PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-LG, cs.CE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.11833v1)  

---


**ABSTRACT**  
Physics-Informed Neural Networks (PINNs) have emerged as a promising deep learning framework for approximating numerical solutions for partial differential equations (PDEs). While conventional PINNs and most related studies adopt fully-connected multilayer perceptrons (MLP) as the backbone structure, they have neglected the temporal relations in PDEs and failed to approximate the true solution. In this paper, we propose a novel Transformer-based framework, namely PINNsFormer, that accurately approximates PDEs' solutions by capturing the temporal dependencies with multi-head attention mechanisms in Transformer-based models. Instead of approximating point predictions, PINNsFormer adapts input vectors to pseudo sequences and point-wise PINNs loss to a sequential PINNs loss. In addition, PINNsFormer is equipped with a novel activation function, namely Wavelet, which anticipates the Fourier decomposition through deep neural networks. We empirically demonstrate PINNsFormer's ability to capture the PDE solutions for various scenarios, in which conventional PINNs have failed to learn. We also show that PINNsFormer achieves superior approximation accuracy on such problems than conventional PINNs with non-sensitive hyperparameters, in trade of marginal computational and memory costs, with extensive experiments.

{{</citation>}}


## cs.DC (1)



### (76/84) A Reinforcement Learning Framework with Region-Awareness and Shared Path Experience for Efficient Routing in Networks-on-Chip (Kamil Khan et al., 2023)

{{<citation>}}

Kamil Khan, Sudeep Pasricha. (2023)  
**A Reinforcement Learning Framework with Region-Awareness and Shared Path Experience for Efficient Routing in Networks-on-Chip**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11712v1)  

---


**ABSTRACT**  
Network-on-chip (NoC) architectures provide a scalable, high-performance, and reliable interconnect for emerging manycore systems. The routing policies used in NoCs have a significant impact on overall performance. Prior efforts have proposed reinforcement learning (RL)-based adaptive routing policies to avoid congestion and minimize latency in NoCs. The output quality of RL policies depends on selecting a representative cost function and an effective update mechanism. Unfortunately, existing RL policies for NoC routing fail to represent path contention and regional congestion in the cost function. Moreover, the experience of packet flows sharing the same route is not fully incorporated into the RL update mechanism. In this paper, we present a novel regional congestion-aware RL-based NoC routing policy called Q-RASP that is capable of sharing experience from packets using the same routes. Q-RASP improves average packet latency by up to 18.3% and reduces NoC energy consumption by up to 6.7% with minimal area overheads compared to state-of-the-art RL-based NoC routing implementations.

{{</citation>}}


## cs.HC (2)



### (77/84) How do you feel? Measuring User-Perceived Value for Rejecting Machine Decisions in Hate Speech Detection (Philippe Lammerts et al., 2023)

{{<citation>}}

Philippe Lammerts, Philip Lippmann, Yen-Chia Hsu, Fabio Casati, Jie Yang. (2023)  
**How do you feel? Measuring User-Perceived Value for Rejecting Machine Decisions in Hate Speech Detection**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2307.11806v1)  

---


**ABSTRACT**  
Hate speech moderation remains a challenging task for social media platforms. Human-AI collaborative systems offer the potential to combine the strengths of humans' reliability and the scalability of machine learning to tackle this issue effectively. While methods for task handover in human-AI collaboration exist that consider the costs of incorrect predictions, insufficient attention has been paid to accurately estimating these costs. In this work, we propose a value-sensitive rejection mechanism that automatically rejects machine decisions for human moderation based on users' value perceptions regarding machine decisions. We conduct a crowdsourced survey study with 160 participants to evaluate their perception of correct and incorrect machine decisions in the domain of hate speech detection, as well as occurrences where the system rejects making a prediction. Here, we introduce Magnitude Estimation, an unbounded scale, as the preferred method for measuring user (dis)agreement with machine decisions. Our results show that Magnitude Estimation can provide a reliable measurement of participants' perception of machine decisions. By integrating user-perceived value into human-AI collaboration, we further show that it can guide us in 1) determining when to accept or reject machine decisions to obtain the optimal total value a model can deliver and 2) selecting better classification models as compared to the more widely used target of model accuracy.

{{</citation>}}


### (78/84) Large Language Model-based System to Provide Immediate Feedback to Students in Flipped Classroom Preparation Learning (Shintaro Uchiyama et al., 2023)

{{<citation>}}

Shintaro Uchiyama, Kyoji Umemura, Yusuke Morita. (2023)  
**Large Language Model-based System to Provide Immediate Feedback to Students in Flipped Classroom Preparation Learning**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.11388v1)  

---


**ABSTRACT**  
This paper proposes a system that uses large language models to provide immediate feedback to students in flipped classroom preparation learning. This study aimed to solve challenges in the flipped classroom model, such as ensuring that students are emotionally engaged and motivated to learn. Students often have questions about the content of lecture videos in the preparation of flipped classrooms, but it is difficult for teachers to answer them immediately. The proposed system was developed using the ChatGPT API on a video-watching support system for preparation learning that is being used in real practice. Answers from ChatGPT often do not align with the context of the student's question. Therefore, this paper also proposes a method to align the answer with the context. This paper also proposes a method to collect the teacher's answers to the students' questions and use them as additional guides for the students. This paper discusses the design and implementation of the proposed system.

{{</citation>}}


## cs.SD (1)



### (79/84) A Change of Heart: Improving Speech Emotion Recognition through Speech-to-Text Modality Conversion (Zeinab Sadat Taghavi et al., 2023)

{{<citation>}}

Zeinab Sadat Taghavi, Ali Satvaty, Hossein Sameti. (2023)  
**A Change of Heart: Improving Speech Emotion Recognition through Speech-to-Text Modality Conversion**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2307.11584v1)  

---


**ABSTRACT**  
Speech Emotion Recognition (SER) is a challenging task. In this paper, we introduce a modality conversion concept aimed at enhancing emotion recognition performance on the MELD dataset. We assess our approach through two experiments: first, a method named Modality-Conversion that employs automatic speech recognition (ASR) systems, followed by a text classifier; second, we assume perfect ASR output and investigate the impact of modality conversion on SER, this method is called Modality-Conversion++. Our findings indicate that the first method yields substantial results, while the second method outperforms state-of-the-art (SOTA) speech-based approaches in terms of SER weighted-F1 (WF1) score on the MELD dataset. This research highlights the potential of modality conversion for tasks that can be conducted in alternative modalities.

{{</citation>}}


## econ.GN (1)



### (80/84) Predict-AI-bility of how humans balance self-interest with the interest of others (Valerio Capraro et al., 2023)

{{<citation>}}

Valerio Capraro, Roberto Di Paolo, Veronica Pizziol. (2023)  
**Predict-AI-bility of how humans balance self-interest with the interest of others**  

---
Primary Category: econ.GN  
Categories: cs-AI, cs-CY, cs-GT, econ-GN, econ.GN, q-fin-EC  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.12776v1)  

---


**ABSTRACT**  
Generative artificial intelligence holds enormous potential to revolutionize decision-making processes, from everyday to high-stake scenarios. However, as many decisions carry social implications, for AI to be a reliable assistant for decision-making it is crucial that it is able to capture the balance between self-interest and the interest of others. We investigate the ability of three of the most advanced chatbots to predict dictator game decisions across 78 experiments with human participants from 12 countries. We find that only GPT-4 (not Bard nor Bing) correctly captures qualitative behavioral patterns, identifying three major classes of behavior: self-interested, inequity-averse, and fully altruistic. Nonetheless, GPT-4 consistently overestimates other-regarding behavior, inflating the proportion of inequity-averse and fully altruistic participants. This bias has significant implications for AI developers and users.

{{</citation>}}


## cs.MA (1)



### (81/84) Providing personalized Explanations: a Conversational Approach (Jieting Luo et al., 2023)

{{<citation>}}

Jieting Luo, Thomas Studer, Mehdi Dastani. (2023)  
**Providing personalized Explanations: a Conversational Approach**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LO, cs-MA, cs.MA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.11452v1)  

---


**ABSTRACT**  
The increasing applications of AI systems require personalized explanations for their behaviors to various stakeholders since the stakeholders may have various knowledge and backgrounds. In general, a conversation between explainers and explainees not only allows explainers to obtain the explainees' background, but also allows explainees to better understand the explanations. In this paper, we propose an approach for an explainer to communicate personalized explanations to an explainee through having consecutive conversations with the explainee. We prove that the conversation terminates due to the explainee's justification of the initial claim as long as there exists an explanation for the initial claim that the explainee understands and the explainer is aware of.

{{</citation>}}


## eess.AS (1)



### (82/84) Prompting Large Language Models with Speech Recognition Abilities (Yassir Fathullah et al., 2023)

{{<citation>}}

Yassir Fathullah, Chunyang Wu, Egor Lakomkin, Junteng Jia, Yuan Shangguan, Ke Li, Jinxi Guo, Wenhan Xiong, Jay Mahadeokar, Ozlem Kalinli, Christian Fuegen, Mike Seltzer. (2023)  
**Prompting Large Language Models with Speech Recognition Abilities**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-LG, eess-AS, eess.AS  
Keywords: LLaMA, Language Model, Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.11795v1)  

---


**ABSTRACT**  
Large language models have proven themselves highly flexible, able to solve a wide range of generative tasks, such as abstractive summarization and open-ended question answering. In this paper we extend the capabilities of LLMs by directly attaching a small audio encoder allowing it to perform speech recognition. By directly prepending a sequence of audial embeddings to the text token embeddings, the LLM can be converted to an automatic speech recognition (ASR) system, and be used in the exact same manner as its textual counterpart. Experiments on Multilingual LibriSpeech (MLS) show that incorporating a conformer encoder into the open sourced LLaMA-7B allows it to outperform monolingual baselines by 18% and perform multilingual speech recognition despite LLaMA being trained overwhelmingly on English text. Furthermore, we perform ablation studies to investigate whether the LLM can be completely frozen during training to maintain its original capabilities, scaling up the audio encoder, and increasing the audio encoder striding to generate fewer embeddings. The results from these studies show that multilingual ASR is possible even when the LLM is frozen or when strides of almost 1 second are used in the audio encoder opening up the possibility for LLMs to operate on long-form audio.

{{</citation>}}


## cs.IT (1)



### (83/84) Attention to Entropic Communication (Torsten Enßlin et al., 2023)

{{<citation>}}

Torsten Enßlin, Carolin Weidinger, Philipp Frank. (2023)  
**Attention to Entropic Communication**  

---
Primary Category: cs.IT  
Categories: 94-10 (Primary) 60Axx (Secondary), I-2-0; I-2-4; I-2-6, cs-IT, cs-LG, cs.IT, math-IT, physics-data-an, stat-ML  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.11423v1)  

---


**ABSTRACT**  
The concept of attention, numerical weights that emphasize the importance of particular data, has proven to be very relevant in artificial intelligence. Relative entropy (RE, aka Kullback-Leibler divergence) plays a central role in communication theory. Here we combine these concepts, attention and RE. RE guides optimal encoding of messages in bandwidth-limited communication as well as optimal message decoding via the maximum entropy principle (MEP). In the coding scenario, RE can be derived from four requirements, namely being analytical, local, proper, and calibrated. Weighted RE, used for attention steering in communications, turns out to be improper. To see how proper attention communication can emerge, we analyze a scenario of a message sender who wants to ensure that the receiver of the message can perform well-informed actions. If the receiver decodes the message using the MEP, the sender only needs to know the receiver's utility function to inform optimally, but not the receiver's initial knowledge state. In case only the curvature of the utility function maxima are known, it becomes desirable to accurately communicate an attention function, in this case a by this curvature weighted and re-normalized probability function. Entropic attention communication is here proposed as the desired generalization of entropic communication that permits weighting while being proper, thereby aiding the design of optimal communication protocols in technical applications and helping to understand human communication. For example, our analysis shows how to derive the level of cooperation expected under misaligned interests of otherwise honest communication partners.

{{</citation>}}


## physics.optics (1)



### (84/84) Artificial Intelligence-Generated Terahertz Multi-Resonant Metasurfaces via Improved Transformer and CGAN Neural Networks (Yangpeng Huang et al., 2023)

{{<citation>}}

Yangpeng Huang, Naixing Feng, Yijun Cai. (2023)  
**Artificial Intelligence-Generated Terahertz Multi-Resonant Metasurfaces via Improved Transformer and CGAN Neural Networks**  

---
Primary Category: physics.optics  
Categories: cs-LG, physics-app-ph, physics-optics, physics.optics  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2307.11794v1)  

---


**ABSTRACT**  
It is well known that the inverse design of terahertz (THz) multi-resonant graphene metasurfaces by using traditional deep neural networks (DNNs) has limited generalization ability. In this paper, we propose improved Transformer and conditional generative adversarial neural networks (CGAN) for the inverse design of graphene metasurfaces based upon THz multi-resonant absorption spectra. The improved Transformer can obtain higher accuracy and generalization performance in the StoV (Spectrum to Vector) design compared to traditional multilayer perceptron (MLP) neural networks, while the StoI (Spectrum to Image) design achieved through CGAN can provide more comprehensive information and higher accuracy than the StoV design obtained by MLP. Moreover, the improved CGAN can achieve the inverse design of graphene metasurface images directly from the desired multi-resonant absorption spectra. It is turned out that this work can finish facilitating the design process of artificial intelligence-generated metasurfaces (AIGM), and even provide a useful guide for developing complex THz metasurfaces based on 2D materials using generative neural networks.

{{</citation>}}
