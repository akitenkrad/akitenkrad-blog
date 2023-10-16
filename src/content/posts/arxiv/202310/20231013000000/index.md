---
draft: false
title: "arXiv @ 2023.10.13"
date: 2023-10-13
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.13"
    identifier: arxiv_20231013
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (7)](#csro-7)
- [cs.LG (34)](#cslg-34)
- [cs.CL (46)](#cscl-46)
- [cs.SE (5)](#csse-5)
- [cs.NI (4)](#csni-4)
- [cs.AI (20)](#csai-20)
- [cs.CV (31)](#cscv-31)
- [cs.HC (8)](#cshc-8)
- [quant-ph (3)](#quant-ph-3)
- [cs.AR (1)](#csar-1)
- [cs.IR (4)](#csir-4)
- [cs.CR (5)](#cscr-5)
- [cs.CE (1)](#csce-1)
- [eess.IV (2)](#eessiv-2)
- [cs.SI (2)](#cssi-2)
- [math.AT (1)](#mathat-1)
- [cs.CY (3)](#cscy-3)
- [eess.SP (1)](#eesssp-1)
- [cs.IT (1)](#csit-1)
- [cs.MM (1)](#csmm-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.MA (1)](#csma-1)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.SD (2)](#cssd-2)

## cs.RO (7)



### (1/185) Co-NavGPT: Multi-Robot Cooperative Visual Semantic Navigation using Large Language Models (Bangguo Yu et al., 2023)

{{<citation>}}

Bangguo Yu, Hamidreza Kasaei, Ming Cao. (2023)  
**Co-NavGPT: Multi-Robot Cooperative Visual Semantic Navigation using Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07937v1)  

---


**ABSTRACT**  
In advanced human-robot interaction tasks, visual target navigation is crucial for autonomous robots navigating unknown environments. While numerous approaches have been developed in the past, most are designed for single-robot operations, which often suffer from reduced efficiency and robustness due to environmental complexities. Furthermore, learning policies for multi-robot collaboration are resource-intensive. To address these challenges, we propose Co-NavGPT, an innovative framework that integrates Large Language Models (LLMs) as a global planner for multi-robot cooperative visual target navigation. Co-NavGPT encodes the explored environment data into prompts, enhancing LLMs' scene comprehension. It then assigns exploration frontiers to each robot for efficient target search. Experimental results on Habitat-Matterport 3D (HM3D) demonstrate that Co-NavGPT surpasses existing models in success rates and efficiency without any learning process, demonstrating the vast potential of LLMs in multi-robot collaboration domains. The supplementary video, prompts, and code can be accessed via the following link: \href{https://sites.google.com/view/co-navgpt}{https://sites.google.com/view/co-navgpt}.

{{</citation>}}


### (2/185) NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration (Ajay Sridhar et al., 2023)

{{<citation>}}

Ajay Sridhar, Dhruv Shah, Catherine Glossop, Sergey Levine. (2023)  
**NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07896v1)  

---


**ABSTRACT**  
Robotic learning for navigation in unfamiliar environments needs to provide policies for both task-oriented navigation (i.e., reaching a goal that the robot has located), and task-agnostic exploration (i.e., searching for a goal in a novel setting). Typically, these roles are handled by separate models, for example by using subgoal proposals, planning, or separate navigation strategies. In this paper, we describe how we can train a single unified diffusion policy to handle both goal-directed navigation and goal-agnostic exploration, with the latter providing the ability to search novel environments, and the former providing the ability to reach a user-specified goal once it has been located. We show that this unified policy results in better overall performance when navigating to visually indicated goals in novel environments, as compared to approaches that use subgoal proposals from generative models, or prior methods based on latent variable models. We instantiate our method by using a large-scale Transformer-based policy trained on data from multiple ground robots, with a diffusion model decoder to flexibly handle both goal-conditioned and goal-agnostic navigation. Our experiments, conducted on a real-world mobile robot platform, show effective navigation in unseen environments in comparison with five alternative methods, and demonstrate significant improvements in performance and lower collision rates, despite utilizing smaller models than state-of-the-art approaches. For more videos, code, and pre-trained model checkpoints, see https://general-navigation-models.github.io/nomad/

{{</citation>}}


### (3/185) Active Learning with Dual Model Predictive Path-Integral Control for Interaction-Aware Autonomous Highway On-ramp Merging (Jacob Knaup et al., 2023)

{{<citation>}}

Jacob Knaup, Jovin D'sa, Behdad Chalaki, Tyler Naes, Hossein Nourkhiz Mahjoub, Ehsan Moradi-Pari, Panagiotis Tsiotras. (2023)  
**Active Learning with Dual Model Predictive Path-Integral Control for Interaction-Aware Autonomous Highway On-ramp Merging**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO, math-OC  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.07840v1)  

---


**ABSTRACT**  
Merging into dense highway traffic for an autonomous vehicle is a complex decision-making task, wherein the vehicle must identify a potential gap and coordinate with surrounding human drivers, each of whom may exhibit diverse driving behaviors. Many existing methods consider other drivers to be dynamic obstacles and, as a result, are incapable of capturing the full intent of the human drivers via this passive planning. In this paper, we propose a novel dual control framework based on Model Predictive Path-Integral control to generate interactive trajectories. This framework incorporates a Bayesian inference approach to actively learn the agents' parameters, i.e., other drivers' model parameters. The proposed framework employs a sampling-based approach that is suitable for real-time implementation through the utilization of GPUs. We illustrate the effectiveness of our proposed methodology through comprehensive numerical simulations conducted in both high and low-fidelity simulation scenarios focusing on autonomous on-ramp merging.

{{</citation>}}


### (4/185) ViT-A*: Legged Robot Path Planning using Vision Transformer A* (Jianwei Liu et al., 2023)

{{<citation>}}

Jianwei Liu, Shirui Lyu, Denis Hadjivelichkov, Valerio Modugno, Dimitrios Kanoulas. (2023)  
**ViT-A*: Legged Robot Path Planning using Vision Transformer A***  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07525v1)  

---


**ABSTRACT**  
Legged robots, particularly quadrupeds, offer promising navigation capabilities, especially in scenarios requiring traversal over diverse terrains and obstacle avoidance. This paper addresses the challenge of enabling legged robots to navigate complex environments effectively through the integration of data-driven path-planning methods. We propose an approach that utilizes differentiable planners, allowing the learning of end-to-end global plans via a neural network for commanding quadruped robots. The approach leverages 2D maps and obstacle specifications as inputs to generate a global path. To enhance the functionality of the developed neural network-based path planner, we use Vision Transformers (ViT) for map pre-processing, to enable the effective handling of larger maps. Experimental evaluations on two real robotic quadrupeds (Boston Dynamics Spot and Unitree Go1) demonstrate the effectiveness and versatility of the proposed approach in generating reliable path plans.

{{</citation>}}


### (5/185) Terrain-adaptive Central Pattern Generators with Reinforcement Learning for Hexapod Locomotion (Qiyue Yang et al., 2023)

{{<citation>}}

Qiyue Yang, Yue Gao, Shaoyuan Li. (2023)  
**Terrain-adaptive Central Pattern Generators with Reinforcement Learning for Hexapod Locomotion**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07744v1)  

---


**ABSTRACT**  
Inspired by biological motion generation, central pattern generators (CPGs) is frequently employed in legged robot locomotion control to produce natural gait pattern with low-dimensional control signals. However, the limited adaptability and stability over complex terrains hinder its application. To address this issue, this paper proposes a terrain-adaptive locomotion control method that incorporates deep reinforcement learning (DRL) framework into CPG, where the CPG model is responsible for the generation of synchronized signals, providing basic locomotion gait, while DRL is integrated to enhance the adaptability of robot towards uneven terrains by adjusting the parameters of CPG mapping functions. The experiments conducted on the hexapod robot in Isaac Gym simulation environment demonstrated the superiority of the proposed method in terrain-adaptability, convergence rate and reward design complexity.

{{</citation>}}


### (6/185) RANS: Highly-Parallelised Simulator for Reinforcement Learning based Autonomous Navigating Spacecrafts (Matteo El-Hariry et al., 2023)

{{<citation>}}

Matteo El-Hariry, Antoine Richard, Miguel Olivares-Mendez. (2023)  
**RANS: Highly-Parallelised Simulator for Reinforcement Learning based Autonomous Navigating Spacecrafts**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07393v1)  

---


**ABSTRACT**  
Nowadays, realistic simulation environments are essential to validate and build reliable robotic solutions. This is particularly true when using Reinforcement Learning (RL) based control policies. To this end, both robotics and RL developers need tools and workflows to create physically accurate simulations and synthetic datasets. Gazebo, MuJoCo, Webots, Pybullets or Isaac Sym are some of the many tools available to simulate robotic systems. Developing learning-based methods for space navigation is, due to the highly complex nature of the problem, an intensive data-driven process that requires highly parallelized simulations. When it comes to the control of spacecrafts, there is no easy to use simulation library designed for RL. We address this gap by harnessing the capabilities of NVIDIA Isaac Gym, where both physics simulation and the policy training reside on GPU. Building on this tool, we provide an open-source library enabling users to simulate thousands of parallel spacecrafts, that learn a set of maneuvering tasks, such as position, attitude, and velocity control. These tasks enable to validate complex space scenarios, such as trajectory optimization for landing, docking, rendezvous and more.

{{</citation>}}


### (7/185) CoPAL: Corrective Planning of Robot Actions with Large Language Models (Frank Joublin et al., 2023)

{{<citation>}}

Frank Joublin, Antonello Ceravola, Pavel Smirnov, Felix Ocker, Joerg Deigmoeller, Anna Belardinelli, Chao Wang, Stephan Hasler, Daniel Tanneberg, Michael Gienger. (2023)  
**CoPAL: Corrective Planning of Robot Actions with Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07263v1)  

---


**ABSTRACT**  
In the pursuit of fully autonomous robotic systems capable of taking over tasks traditionally performed by humans, the complexity of open-world environments poses a considerable challenge. Addressing this imperative, this study contributes to the field of Large Language Models (LLMs) applied to task and motion planning for robots. We propose a system architecture that orchestrates a seamless interplay between multiple cognitive levels, encompassing reasoning, planning, and motion generation. At its core lies a novel replanning strategy that handles physically grounded, logical, and semantic errors in the generated plans. We demonstrate the efficacy of the proposed feedback architecture, particularly its impact on executability, correctness, and time complexity via empirical evaluation in the context of a simulation and two intricate real-world scenarios: blocks world, barman and pizza preparation.

{{</citation>}}


## cs.LG (34)



### (8/185) D2 Pruning: Message Passing for Balancing Diversity and Difficulty in Data Pruning (Adyasha Maharana et al., 2023)

{{<citation>}}

Adyasha Maharana, Prateek Yadav, Mohit Bansal. (2023)  
**D2 Pruning: Message Passing for Balancing Diversity and Difficulty in Data Pruning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2310.07931v1)  

---


**ABSTRACT**  
Analytical theories suggest that higher-quality data can lead to lower test errors in models trained on a fixed data budget. Moreover, a model can be trained on a lower compute budget without compromising performance if a dataset can be stripped of its redundancies. Coreset selection (or data pruning) seeks to select a subset of the training data so as to maximize the performance of models trained on this subset, also referred to as coreset. There are two dominant approaches: (1) geometry-based data selection for maximizing data diversity in the coreset, and (2) functions that assign difficulty scores to samples based on training dynamics. Optimizing for data diversity leads to a coreset that is biased towards easier samples, whereas, selection by difficulty ranking omits easy samples that are necessary for the training of deep learning models. This demonstrates that data diversity and importance scores are two complementary factors that need to be jointly considered during coreset selection. We represent a dataset as an undirected graph and propose a novel pruning algorithm, D2 Pruning, that uses forward and reverse message passing over this dataset graph for coreset selection. D2 Pruning updates the difficulty scores of each example by incorporating the difficulty of its neighboring examples in the dataset graph. Then, these updated difficulty scores direct a graph-based sampling method to select a coreset that encapsulates both diverse and difficult regions of the dataset space. We evaluate supervised and self-supervised versions of our method on various vision and language datasets. Results show that D2 Pruning improves coreset selection over previous state-of-the-art methods for up to 70% pruning rates. Additionally, we find that using D2 Pruning for filtering large multimodal datasets leads to increased diversity in the dataset and improved generalization of pretrained models.

{{</citation>}}


### (9/185) The Expresssive Power of Transformers with Chain of Thought (William Merrill et al., 2023)

{{<citation>}}

William Merrill, Ashish Sabharwal. (2023)  
**The Expresssive Power of Transformers with Chain of Thought**  

---
Primary Category: cs.LG  
Categories: cs-CC, cs-CL, cs-LG, cs-LO, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07923v1)  

---


**ABSTRACT**  
Recent theoretical work has identified surprisingly simple reasoning problems, such as checking if two nodes in a graph are connected or simulating finite-state machines, that are provably unsolvable by standard transformers that answer immediately after reading their input. However, in practice, transformers' reasoning can be improved by allowing them to use a "chain of thought" or "scratchpad", i.e., generate and condition on a sequence of intermediate tokens before answering. Motivated by this, we ask: Does such intermediate generation fundamentally extend the computational power of a decoder-only transformer? We show that the answer is yes, but the amount of increase depends crucially on the amount of intermediate generation. For instance, we find that transformer decoders with a logarithmic number of decoding steps (w.r.t. the input length) push the limits of standard transformers only slightly, while a linear number of decoding steps adds a clear new ability (under standard complexity conjectures): recognizing all regular languages. Our results also imply that linear steps keep transformer decoders within context-sensitive languages, and polynomial steps make them recognize exactly the class of polynomial-time solvable problems -- the first exact characterization of a type of transformers in terms of standard complexity classes. Together, our results provide a nuanced framework for understanding how the length of a transformer's chain of thought or scratchpad impacts its reasoning power.

{{</citation>}}


### (10/185) Leader-Follower Neural Networks with Local Error Signals Inspired by Complex Collectives (Chenzhong Yin et al., 2023)

{{<citation>}}

Chenzhong Yin, Mingxi Cheng, Xiongye Xiao, Xinghe Chen, Shahin Nazarian, Andrei Irimia, Paul Bogdan. (2023)  
**Leader-Follower Neural Networks with Local Error Signals Inspired by Complex Collectives**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.07885v1)  

---


**ABSTRACT**  
The collective behavior of a network with heterogeneous, resource-limited information processing units (e.g., group of fish, flock of birds, or network of neurons) demonstrates high self-organization and complexity. These emergent properties arise from simple interaction rules where certain individuals can exhibit leadership-like behavior and influence the collective activity of the group. Motivated by the intricacy of these collectives, we propose a neural network (NN) architecture inspired by the rules observed in nature's collective ensembles. This NN structure contains workers that encompass one or more information processing units (e.g., neurons, filters, layers, or blocks of layers). Workers are either leaders or followers, and we train a leader-follower neural network (LFNN) by leveraging local error signals and optionally incorporating backpropagation (BP) and global loss. We investigate worker behavior and evaluate LFNNs through extensive experimentation. Our LFNNs trained with local error signals achieve significantly lower error rates than previous BP-free algorithms on MNIST and CIFAR-10 and even surpass BP-enabled baselines. In the case of ImageNet, our LFNN-l demonstrates superior scalability and outperforms previous BP-free algorithms by a significant margin.

{{</citation>}}


### (11/185) The Thousand Faces of Explainable AI Along the Machine Learning Life Cycle: Industrial Reality and Current State of Research (Thomas Decker et al., 2023)

{{<citation>}}

Thomas Decker, Ralf Gross, Alexander Koebler, Michael Lebacher, Ronald Schnitzer, Stefan H. Weber. (2023)  
**The Thousand Faces of Explainable AI Along the Machine Learning Life Cycle: Industrial Reality and Current State of Research**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07882v1)  

---


**ABSTRACT**  
In this paper, we investigate the practical relevance of explainable artificial intelligence (XAI) with a special focus on the producing industries and relate them to the current state of academic XAI research. Our findings are based on an extensive series of interviews regarding the role and applicability of XAI along the Machine Learning (ML) lifecycle in current industrial practice and its expected relevance in the future. The interviews were conducted among a great variety of roles and key stakeholders from different industry sectors. On top of that, we outline the state of XAI research by providing a concise review of the relevant literature. This enables us to provide an encompassing overview covering the opinions of the surveyed persons as well as the current state of academic research. By comparing our interview results with the current research approaches we reveal several discrepancies. While a multitude of different XAI approaches exists, most of them are centered around the model evaluation phase and data scientists. Their versatile capabilities for other stages are currently either not sufficiently explored or not popular among practitioners. In line with existing work, our findings also confirm that more efforts are needed to enable also non-expert users' interpretation and understanding of opaque AI models with existing methods and frameworks.

{{</citation>}}


### (12/185) Measuring Feature Sparsity in Language Models (Mingyang Deng et al., 2023)

{{<citation>}}

Mingyang Deng, Lucas Tao, Joe Benton. (2023)  
**Measuring Feature Sparsity in Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07837v1)  

---


**ABSTRACT**  
Recent works have proposed that activations in language models can be modelled as sparse linear combinations of vectors corresponding to features of input text. Under this assumption, these works aimed to reconstruct feature directions using sparse coding. We develop metrics to assess the success of these sparse coding techniques and test the validity of the linearity and sparsity assumptions. We show our metrics can predict the level of sparsity on synthetic sparse linear activations, and can distinguish between sparse linear data and several other distributions. We use our metrics to measure levels of sparsity in several language models. We find evidence that language model activations can be accurately modelled by sparse linear combinations of features, significantly more so than control datasets. We also show that model activations appear to be sparsest in the first and final layers.

{{</citation>}}


### (13/185) Large Language Models Are Zero-Shot Time Series Forecasters (Nate Gruver et al., 2023)

{{<citation>}}

Nate Gruver, Marc Finzi, Shikai Qiu, Andrew Gordon Wilson. (2023)  
**Large Language Models Are Zero-Shot Time Series Forecasters**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GPT, GPT-4, LLaMA, Language Model, Time Series, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.07820v1)  

---


**ABSTRACT**  
By encoding time series as a string of numerical digits, we can frame time series forecasting as next-token prediction in text. Developing this approach, we find that large language models (LLMs) such as GPT-3 and LLaMA-2 can surprisingly zero-shot extrapolate time series at a level comparable to or exceeding the performance of purpose-built time series models trained on the downstream tasks. To facilitate this performance, we propose procedures for effectively tokenizing time series data and converting discrete distributions over tokens into highly flexible densities over continuous values. We argue the success of LLMs for time series stems from their ability to naturally represent multimodal distributions, in conjunction with biases for simplicity, and repetition, which align with the salient features in many time series, such as repeated seasonal trends. We also show how LLMs can naturally handle missing data without imputation through non-numerical text, accommodate textual side information, and answer questions to help explain predictions. While we find that increasing model size generally improves performance on time series, we show GPT-4 can perform worse than GPT-3 because of how it tokenizes numbers, and poor uncertainty calibration, which is likely the result of alignment interventions such as RLHF.

{{</citation>}}


### (14/185) Self-supervised Representation Learning From Random Data Projectors (Yi Sui et al., 2023)

{{<citation>}}

Yi Sui, Tongzi Wu, Jesse C. Cresswell, Ga Wu, George Stein, Xiao Shi Huang, Xiaochen Zhang, Maksims Volkovs. (2023)  
**Self-supervised Representation Learning From Random Data Projectors**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.07756v1)  

---


**ABSTRACT**  
Self-supervised representation learning~(SSRL) has advanced considerably by exploiting the transformation invariance assumption under artificially designed data augmentations. While augmentation-based SSRL algorithms push the boundaries of performance in computer vision and natural language processing, they are often not directly applicable to other data modalities, and can conflict with application-specific data augmentation constraints. This paper presents an SSRL approach that can be applied to any data modality and network architecture because it does not rely on augmentations or masking. Specifically, we show that high-quality data representations can be learned by reconstructing random data projections. We evaluate the proposed approach on a wide range of representation learning tasks that span diverse modalities and real-world applications. We show that it outperforms multiple state-of-the-art SSRL baselines. Due to its wide applicability and strong empirical results, we argue that learning from randomness is a fruitful research direction worthy of attention and further study.

{{</citation>}}


### (15/185) MatFormer: Nested Transformer for Elastic Inference (Devvrit et al., 2023)

{{<citation>}}

Devvrit, Sneha Kudugunta, Aditya Kusupati, Tim Dettmers, Kaifeng Chen, Inderjit Dhillon, Yulia Tsvetkov, Hannaneh Hajishirzi, Sham Kakade, Ali Farhadi, Prateek Jain. (2023)  
**MatFormer: Nested Transformer for Elastic Inference**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: PaLM, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07707v1)  

---


**ABSTRACT**  
Transformer models are deployed in a wide range of settings, from multi-accelerator clusters to standalone mobile phones. The diverse inference constraints in these scenarios necessitate practitioners to train foundation models such as PaLM 2, Llama, & ViTs as a series of models of varying sizes. Due to significant training costs, only a select few model sizes are trained and supported, limiting more fine-grained control over relevant tradeoffs, including latency, cost, and accuracy. This work introduces MatFormer, a nested Transformer architecture designed to offer elasticity in a variety of deployment constraints. Each Feed Forward Network (FFN) block of a MatFormer model is jointly optimized with a few nested smaller FFN blocks. This training procedure allows for the Mix'n'Match of model granularities across layers -- i.e., a trained universal MatFormer model enables extraction of hundreds of accurate smaller models, which were never explicitly optimized. We empirically demonstrate MatFormer's effectiveness across different model classes (decoders & encoders), modalities (language & vision), and scales (up to 2.6B parameters). We find that a 2.6B decoder-only MatFormer language model (MatLM) allows us to extract smaller models spanning from 1.5B to 2.6B, each exhibiting comparable validation loss and one-shot downstream evaluations to their independently trained counterparts. Furthermore, we observe that smaller encoders extracted from a universal MatFormer-based ViT (MatViT) encoder preserve the metric-space structure for adaptive large-scale retrieval. Finally, we showcase that speculative decoding with the accurate and consistent submodels extracted from MatFormer can further reduce inference latency.

{{</citation>}}


### (16/185) Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples (Hao Sun et al., 2023)

{{<citation>}}

Hao Sun, Alihan Hüyük, Daniel Jarrett, Mihaela van der Schaar. (2023)  
**Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07747v1)  

---


**ABSTRACT**  
Learning transparent, interpretable controllers with offline data in decision-making systems is an essential area of research due to its potential to reduce the risk of applications in real-world systems. However, in responsibility-sensitive settings such as healthcare, decision accountability is of paramount importance, yet has not been adequately addressed by the literature. This paper introduces the Accountable Offline Controller (AOC) that employs the offline dataset as the Decision Corpus and performs accountable control based on a tailored selection of examples, referred to as the Corpus Subset. ABC operates effectively in low-data scenarios, can be extended to the strictly offline imitation setting, and displays qualities of both conservation and adaptability. We assess ABC's performance in both simulated and real-world healthcare scenarios, emphasizing its capability to manage offline control tasks with high levels of performance while maintaining accountability.   Keywords: Interpretable Reinforcement Learning, Explainable Reinforcement Learning, Reinforcement Learning Transparency, Offline Reinforcement Learning, Batched Control.

{{</citation>}}


### (17/185) GRaMuFeN: Graph-based Multi-modal Fake News Detection in Social Media (Makan Kananian et al., 2023)

{{<citation>}}

Makan Kananian, Fatima Badiei, S. AmirAli Gh. Ghahramani. (2023)  
**GRaMuFeN: Graph-based Multi-modal Fake News Detection in Social Media**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Fake News, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2310.07668v1)  

---


**ABSTRACT**  
The proliferation of social media platforms such as Twitter, Instagram, and Weibo has significantly enhanced the dissemination of false information. This phenomenon grants both individuals and governmental entities the ability to shape public opinions, highlighting the need for deploying effective detection methods. In this paper, we propose GraMuFeN, a model designed to detect fake content by analyzing both the textual and image content of news. GraMuFeN comprises two primary components: a text encoder and an image encoder. For textual analysis, GraMuFeN treats each text as a graph and employs a Graph Convolutional Neural Network (GCN) as the text encoder. Additionally, the pre-trained ResNet-152, as a Convolutional Neural Network (CNN), has been utilized as the image encoder. By integrating the outputs from these two encoders and implementing a contrastive similarity loss function, GraMuFeN achieves remarkable results. Extensive evaluations conducted on two publicly available benchmark datasets for social media news indicate a 10 % increase in micro F1-Score, signifying improvement over existing state-of-the-art models. These findings underscore the effectiveness of combining GCN and CNN models for detecting fake news in multi-modal data, all while minimizing the additional computational burden imposed by model parameters.

{{</citation>}}


### (18/185) Global Minima, Recoverability Thresholds, and Higher-Order Structure in GNNS (Drake Brown et al., 2023)

{{<citation>}}

Drake Brown, Trevor Garrity, Kaden Parker, Jason Oliphant, Stone Carson, Cole Hanson, Zachary Boyd. (2023)  
**Global Minima, Recoverability Thresholds, and Higher-Order Structure in GNNS**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07667v1)  

---


**ABSTRACT**  
We analyze the performance of graph neural network (GNN) architectures from the perspective of random graph theory. Our approach promises to complement existing lenses on GNN analysis, such as combinatorial expressive power and worst-case adversarial analysis, by connecting the performance of GNNs to typical-case properties of the training data. First, we theoretically characterize the nodewise accuracy of one- and two-layer GCNs relative to the contextual stochastic block model (cSBM) and related models. We additionally prove that GCNs cannot beat linear models under certain circumstances. Second, we numerically map the recoverability thresholds, in terms of accuracy, of four diverse GNN architectures (GCN, GAT, SAGE, and Graph Transformer) under a variety of assumptions about the data. Sample results of this second analysis include: heavy-tailed degree distributions enhance GNN performance, GNNs can work well on strongly heterophilous graphs, and SAGE and Graph Transformer can perform well on arbitrarily noisy edge data, but no architecture handled sufficiently noisy feature data well. Finally, we show how both specific higher-order structures in synthetic data and the mix of empirical structures in real data have dramatic effects (usually negative) on GNN performance.

{{</citation>}}


### (19/185) Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey (Gregory Palmer et al., 2023)

{{<citation>}}

Gregory Palmer, Chris Parry, Daniel J. B. Harrold, Chris Willis. (2023)  
**Deep Reinforcement Learning for Autonomous Cyber Operations: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07745v1)  

---


**ABSTRACT**  
The rapid increase in the number of cyber-attacks in recent years raises the need for principled methods for defending networks against malicious actors. Deep reinforcement learning (DRL) has emerged as a promising approach for mitigating these attacks. However, while DRL has shown much potential for cyber-defence, numerous challenges must be overcome before DRL can be applied to autonomous cyber-operations (ACO) at scale. Principled methods are required for environments that confront learners with very high-dimensional state spaces, large multi-discrete action spaces, and adversarial learning. Recent works have reported success in solving these problems individually. There have also been impressive engineering efforts towards solving all three for real-time strategy games. However, applying DRL to the full ACO problem remains an open challenge. Here, we survey the relevant DRL literature and conceptualize an idealised ACO-DRL agent. We provide: i.) A summary of the domain properties that define the ACO problem; ii.) A comprehensive evaluation of the extent to which domains used for benchmarking DRL approaches are comparable to ACO; iii.) An overview of state-of-the-art approaches for scaling DRL to domains that confront learners with the curse of dimensionality, and; iv.) A survey and critique of current methods for limiting the exploitability of agents within adversarial settings from the perspective of ACO. We conclude with open research questions that we hope will motivate future directions for researchers and practitioners working on ACO.

{{</citation>}}


### (20/185) Graph Transformer Network for Flood Forecasting with Heterogeneous Covariates (Jimeng Shi et al., 2023)

{{<citation>}}

Jimeng Shi, Vitalii Stebliankin, Zhaonan Wang, Shaowen Wang, Giri Narasimhan. (2023)  
**Graph Transformer Network for Flood Forecasting with Heterogeneous Covariates**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07631v1)  

---


**ABSTRACT**  
Floods can be very destructive causing heavy damage to life, property, and livelihoods. Global climate change and the consequent sea-level rise have increased the occurrence of extreme weather events, resulting in elevated and frequent flood risk. Therefore, accurate and timely flood forecasting in coastal river systems is critical to facilitate good flood management. However, the computational tools currently used are either slow or inaccurate. In this paper, we propose a Flood prediction tool using Graph Transformer Network (FloodGTN) for river systems. More specifically, FloodGTN learns the spatio-temporal dependencies of water levels at different monitoring stations using Graph Neural Networks (GNNs) and an LSTM. It is currently implemented to consider external covariates such as rainfall, tide, and the settings of hydraulic structures (e.g., outflows of dams, gates, pumps, etc.) along the river. We use a Transformer to learn the attention given to external covariates in computing water levels. We apply the FloodGTN tool to data from the South Florida Water Management District, which manages a coastal area prone to frequent storms and hurricanes. Experimental results show that FloodGTN outperforms the physics-based model (HEC-RAS) by achieving higher accuracy with 70% improvement while speeding up run times by at least 500x.

{{</citation>}}


### (21/185) PHYDI: Initializing Parameterized Hypercomplex Neural Networks as Identity Functions (Matteo Mancanelli et al., 2023)

{{<citation>}}

Matteo Mancanelli, Eleonora Grassucci, Aurelio Uncini, Danilo Comminiello. (2023)  
**PHYDI: Initializing Parameterized Hypercomplex Neural Networks as Identity Functions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-ET, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07612v1)  

---


**ABSTRACT**  
Neural models based on hypercomplex algebra systems are growing and prolificating for a plethora of applications, ranging from computer vision to natural language processing. Hand in hand with their adoption, parameterized hypercomplex neural networks (PHNNs) are growing in size and no techniques have been adopted so far to control their convergence at a large scale. In this paper, we study PHNNs convergence and propose parameterized hypercomplex identity initialization (PHYDI), a method to improve their convergence at different scales, leading to more robust performance when the number of layers scales up, while also reaching the same performance with fewer iterations. We show the effectiveness of this approach in different benchmarks and with common PHNNs with ResNets- and Transformer-based architecture. The code is available at https://github.com/ispamm/PHYDI.

{{</citation>}}


### (22/185) Survey on Imbalanced Data, Representation Learning and SEP Forecasting (Josias Moukpe, 2023)

{{<citation>}}

Josias Moukpe. (2023)  
**Survey on Imbalanced Data, Representation Learning and SEP Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.07598v1)  

---


**ABSTRACT**  
Deep Learning methods have significantly advanced various data-driven tasks such as regression, classification, and forecasting. However, much of this progress has been predicated on the strong but often unrealistic assumption that training datasets are balanced with respect to the targets they contain. This misalignment with real-world conditions, where data is frequently imbalanced, hampers the effectiveness of such models in practical applications. Methods that reconsider that assumption and tackle real-world imbalances have begun to emerge and explore avenues to address this challenge. One such promising avenue is representation learning, which enables models to capture complex data characteristics and generalize better to minority classes. By focusing on a richer representation of the feature space, these techniques hold the potential to mitigate the impact of data imbalance. In this survey, we present deep learning works that step away from the balanced-data assumption, employing strategies like representation learning to better approximate real-world imbalances. We also highlight a critical application in SEP forecasting where addressing data imbalance is paramount for success.

{{</citation>}}


### (23/185) Transformers for Green Semantic Communication: Less Energy, More Semantics (Shubhabrata Mukherjee et al., 2023)

{{<citation>}}

Shubhabrata Mukherjee, Cory Beard, Sejun Song. (2023)  
**Transformers for Green Semantic Communication: Less Energy, More Semantics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07592v1)  

---


**ABSTRACT**  
Semantic communication aims to transmit meaningful and effective information rather than focusing on individual symbols or bits, resulting in benefits like reduced latency, bandwidth usage, and higher throughput compared to traditional communication. However, semantic communication poses significant challenges due to the need for universal metrics for benchmarking the joint effects of semantic information loss and practical energy consumption. This research presents a novel multi-objective loss function named "Energy-Optimized Semantic Loss" (EOSL), addressing the challenge of balancing semantic information loss and energy consumption. Through comprehensive experiments on transformer models, including CPU and GPU energy usage, it is demonstrated that EOSL-based encoder model selection can save up to 90\% of energy while achieving a 44\% improvement in semantic similarity performance during inference in this experiment. This work paves the way for energy-efficient neural network selection and the development of greener semantic communication architectures.

{{</citation>}}


### (24/185) Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer (Zikai Xiao et al., 2023)

{{<citation>}}

Zikai Xiao, Zihan Chen, Songshang Liu, Hualiang Wang, Yang Feng, Jin Hao, Joey Tianyi Zhou, Jian Wu, Howard Hao Yang, Zuozhu Liu. (2023)  
**Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer**  

---
Primary Category: cs.LG  
Categories: I-2-0, cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.07587v1)  

---


**ABSTRACT**  
Data privacy and long-tailed distribution are the norms rather than the exception in many real-world tasks. This paper investigates a federated long-tailed learning (Fed-LT) task in which each client holds a locally heterogeneous dataset; if the datasets can be globally aggregated, they jointly exhibit a long-tailed distribution. Under such a setting, existing federated optimization and/or centralized long-tailed learning methods hardly apply due to challenges in (a) characterizing the global long-tailed distribution under privacy constraints and (b) adjusting the local learning strategy to cope with the head-tail imbalance. In response, we propose a method termed $\texttt{Fed-GraB}$, comprised of a Self-adjusting Gradient Balancer (SGB) module that re-weights clients' gradients in a closed-loop manner, based on the feedback of global long-tailed distribution evaluated by a Direct Prior Analyzer (DPA) module. Using $\texttt{Fed-GraB}$, clients can effectively alleviate the distribution drift caused by data heterogeneity during the model training process and obtain a global model with better performance on the minority classes while maintaining the performance of the majority classes. Extensive experiments demonstrate that $\texttt{Fed-GraB}$ achieves state-of-the-art performance on representative datasets such as CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and iNaturalist.

{{</citation>}}


### (25/185) Linear Latent World Models in Simple Transformers: A Case Study on Othello-GPT (Dean S. Hazineh et al., 2023)

{{<citation>}}

Dean S. Hazineh, Zechen Zhang, Jeffery Chiu. (2023)  
**Linear Latent World Models in Simple Transformers: A Case Study on Othello-GPT**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07582v2)  

---


**ABSTRACT**  
Foundation models exhibit significant capabilities in decision-making and logical deductions. Nonetheless, a continuing discourse persists regarding their genuine understanding of the world as opposed to mere stochastic mimicry. This paper meticulously examines a simple transformer trained for Othello, extending prior research to enhance comprehension of the emergent world model of Othello-GPT. The investigation reveals that Othello-GPT encapsulates a linear representation of opposing pieces, a factor that causally steers its decision-making process. This paper further elucidates the interplay between the linear world representation and causal decision-making, and their dependence on layer depth and model complexity. We have made the code public.

{{</citation>}}


### (26/185) In-Context Unlearning: Language Models as Few Shot Unlearners (Martin Pawelczyk et al., 2023)

{{<citation>}}

Martin Pawelczyk, Seth Neel, Himabindu Lakkaraju. (2023)  
**In-Context Unlearning: Language Models as Few Shot Unlearners**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07579v2)  

---


**ABSTRACT**  
Machine unlearning, the study of efficiently removing the impact of specific training points on the trained model, has garnered increased attention of late, driven by the need to comply with privacy regulations like the Right to be Forgotten. Although unlearning is particularly relevant for LLMs in light of the copyright issues they raise, achieving precise unlearning is computationally infeasible for very large models. To this end, recent work has proposed several algorithms which approximate the removal of training data without retraining the model. These algorithms crucially rely on access to the model parameters in order to update them, an assumption that may not hold in practice due to computational constraints or when the LLM is accessed via API. In this work, we propose a new class of unlearning methods for LLMs we call ''In-Context Unlearning'', providing inputs in context and without having to update model parameters. To unlearn a particular training instance, we provide the instance alongside a flipped label and additional correctly labelled instances which are prepended as inputs to the LLM at inference time. Our experimental results demonstrate that these contexts effectively remove specific information from the training set while maintaining performance levels that are competitive with (or in some cases exceed) state-of-the-art unlearning methods that require access to the LLM parameters.

{{</citation>}}


### (27/185) Exploiting Causal Graph Priors with Posterior Sampling for Reinforcement Learning (Mirco Mutti et al., 2023)

{{<citation>}}

Mirco Mutti, Riccardo De Santi, Marcello Restelli, Alexander Marx, Giorgia Ramponi. (2023)  
**Exploiting Causal Graph Priors with Posterior Sampling for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07518v1)  

---


**ABSTRACT**  
Posterior sampling allows the exploitation of prior knowledge of the environment's transition dynamics to improve the sample efficiency of reinforcement learning. The prior is typically specified as a class of parametric distributions, a task that can be cumbersome in practice, often resulting in the choice of uninformative priors. In this work, we propose a novel posterior sampling approach in which the prior is given as a (partial) causal graph over the environment's variables. The latter is often more natural to design, such as listing known causal dependencies between biometric features in a medical treatment study. Specifically, we propose a hierarchical Bayesian procedure, called C-PSRL, simultaneously learning the full causal graph at the higher level and the parameters of the resulting factored dynamics at the lower level. For this procedure, we provide an analysis of its Bayesian regret, which explicitly connects the regret rate with the degree of prior knowledge. Our numerical evaluation conducted in illustrative domains confirms that C-PSRL strongly improves the efficiency of posterior sampling with an uninformative prior while performing close to posterior sampling with the full causal graph.

{{</citation>}}


### (28/185) Generalized Mixture Model for Extreme Events Forecasting in Time Series Data (Jincheng Wang et al., 2023)

{{<citation>}}

Jincheng Wang, Yue Gao. (2023)  
**Generalized Mixture Model for Extreme Events Forecasting in Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2310.07435v1)  

---


**ABSTRACT**  
Time Series Forecasting (TSF) is a widely researched topic with broad applications in weather forecasting, traffic control, and stock price prediction. Extreme values in time series often significantly impact human and natural systems, but predicting them is challenging due to their rare occurrence. Statistical methods based on Extreme Value Theory (EVT) provide a systematic approach to modeling the distribution of extremes, particularly the Generalized Pareto (GP) distribution for modeling the distribution of exceedances beyond a threshold. To overcome the subpar performance of deep learning in dealing with heavy-tailed data, we propose a novel framework to enhance the focus on extreme events. Specifically, we propose a Deep Extreme Mixture Model with Autoencoder (DEMMA) for time series prediction. The model comprises two main modules: 1) a generalized mixture distribution based on the Hurdle model and a reparameterized GP distribution form independent of the extreme threshold, 2) an Autoencoder-based LSTM feature extractor and a quantile prediction module with a temporal attention mechanism. We demonstrate the effectiveness of our approach on multiple real-world rainfall datasets.

{{</citation>}}


### (29/185) Non-backtracking Graph Neural Networks (Seonghyun Park et al., 2023)

{{<citation>}}

Seonghyun Park, Narae Ryu, Gahee Kim, Dongyeop Woo, Se-Young Yun, Sungsoo Ahn. (2023)  
**Non-backtracking Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.07430v1)  

---


**ABSTRACT**  
The celebrated message-passing updates for graph neural networks allow the representation of large-scale graphs with local and computationally tractable updates. However, the local updates suffer from backtracking, i.e., a message flows through the same edge twice and revisits the previously visited node. Since the number of message flows increases exponentially with the number of updates, the redundancy in local updates prevents the graph neural network from accurately recognizing a particular message flow for downstream tasks. In this work, we propose to resolve such a redundancy via the non-backtracking graph neural network (NBA-GNN) that updates a message without incorporating the message from the previously visited node. We further investigate how NBA-GNN alleviates the over-squashing of GNNs, and establish a connection between NBA-GNN and the impressive performance of non-backtracking updates for stochastic block model recovery. We empirically verify the effectiveness of our NBA-GNN on long-range graph benchmark and transductive node classification problems.

{{</citation>}}


### (30/185) Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages (Guozheng Ma et al., 2023)

{{<citation>}}

Guozheng Ma, Lu Li, Sen Zhang, Zixuan Liu, Zhen Wang, Yixin Chen, Li Shen, Xueqian Wang, Dacheng Tao. (2023)  
**Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07418v1)  

---


**ABSTRACT**  
Plasticity, the ability of a neural network to evolve with new data, is crucial for high-performance and sample-efficient visual reinforcement learning (VRL). Although methods like resetting and regularization can potentially mitigate plasticity loss, the influences of various components within the VRL framework on the agent's plasticity are still poorly understood. In this work, we conduct a systematic empirical exploration focusing on three primary underexplored facets and derive the following insightful conclusions: (1) data augmentation is essential in maintaining plasticity; (2) the critic's plasticity loss serves as the principal bottleneck impeding efficient training; and (3) without timely intervention to recover critic's plasticity in the early stages, its loss becomes catastrophic. These insights suggest a novel strategy to address the high replay ratio (RR) dilemma, where exacerbated plasticity loss hinders the potential improvements of sample efficiency brought by increased reuse frequency. Rather than setting a static RR for the entire training process, we propose Adaptive RR, which dynamically adjusts the RR based on the critic's plasticity level. Extensive evaluations indicate that Adaptive RR not only avoids catastrophic plasticity loss in the early stages but also benefits from more frequent reuse in later phases, resulting in superior sample efficiency.

{{</citation>}}


### (31/185) NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time Series Pretraining (Chenguo Lin et al., 2023)

{{<citation>}}

Chenguo Lin, Xumeng Wen, Wei Cao, Congrui Huang, Jiang Bian, Stephen Lin, Zhirong Wu. (2023)  
**NuTime: Numerically Multi-Scaled Embedding for Large-Scale Time Series Pretraining**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07402v2)  

---


**ABSTRACT**  
Recent research on time-series self-supervised models shows great promise in learning semantic representations. However, it has been limited to small-scale datasets, e.g., thousands of temporal sequences. In this work, we make key technical contributions that are tailored to the numerical properties of time-series data and allow the model to scale to large datasets, e.g., millions of temporal sequences. We adopt the Transformer architecture by first partitioning the input into non-overlapping windows. Each window is then characterized by its normalized shape and two scalar values denoting the mean and standard deviation within each window. To embed scalar values that may possess arbitrary numerical scales to high-dimensional vectors, we propose a numerically multi-scaled embedding module enumerating all possible scales for the scalar values. The model undergoes pretraining using the proposed numerically multi-scaled embedding with a simple contrastive objective on a large-scale dataset containing over a million sequences. We study its transfer performance on a number of univariate and multivariate classification benchmarks. Our method exhibits remarkable improvement against previous representation learning approaches and establishes the new state of the art, even compared with domain-specific non-learning-based methods.

{{</citation>}}


### (32/185) Histopathological Image Classification and Vulnerability Analysis using Federated Learning (Sankalp Vyas et al., 2023)

{{<citation>}}

Sankalp Vyas, Amar Nath Patra, Raj Mani Shukla. (2023)  
**Histopathological Image Classification and Vulnerability Analysis using Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2310.07380v1)  

---


**ABSTRACT**  
Healthcare is one of the foremost applications of machine learning (ML). Traditionally, ML models are trained by central servers, which aggregate data from various distributed devices to forecast the results for newly generated data. This is a major concern as models can access sensitive user information, which raises privacy concerns. A federated learning (FL) approach can help address this issue: A global model sends its copy to all clients who train these copies, and the clients send the updates (weights) back to it. Over time, the global model improves and becomes more accurate. Data privacy is protected during training, as it is conducted locally on the clients' devices.   However, the global model is susceptible to data poisoning. We develop a privacy-preserving FL technique for a skin cancer dataset and show that the model is prone to data poisoning attacks. Ten clients train the model, but one of them intentionally introduces flipped labels as an attack. This reduces the accuracy of the global model. As the percentage of label flipping increases, there is a noticeable decrease in accuracy. We use a stochastic gradient descent optimization algorithm to find the most optimal accuracy for the model. Although FL can protect user privacy for healthcare diagnostics, it is also vulnerable to data poisoning, which must be addressed.

{{</citation>}}


### (33/185) Atom-Motif Contrastive Transformer for Molecular Property Prediction (Wentao Yu et al., 2023)

{{<citation>}}

Wentao Yu, Shuo Chen, Chen Gong, Gang Niu, Masashi Sugiyama. (2023)  
**Atom-Motif Contrastive Transformer for Molecular Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07351v1)  

---


**ABSTRACT**  
Recently, Graph Transformer (GT) models have been widely used in the task of Molecular Property Prediction (MPP) due to their high reliability in characterizing the latent relationship among graph nodes (i.e., the atoms in a molecule). However, most existing GT-based methods usually explore the basic interactions between pairwise atoms, and thus they fail to consider the important interactions among critical motifs (e.g., functional groups consisted of several atoms) of molecules. As motifs in a molecule are significant patterns that are of great importance for determining molecular properties (e.g., toxicity and solubility), overlooking motif interactions inevitably hinders the effectiveness of MPP. To address this issue, we propose a novel Atom-Motif Contrastive Transformer (AMCT), which not only explores the atom-level interactions but also considers the motif-level interactions. Since the representations of atoms and motifs for a given molecule are actually two different views of the same instance, they are naturally aligned to generate the self-supervisory signals for model training. Meanwhile, the same motif can exist in different molecules, and hence we also employ the contrastive loss to maximize the representation agreement of identical motifs across different molecules. Finally, in order to clearly identify the motifs that are critical in deciding the properties of each molecule, we further construct a property-aware attention mechanism into our learning framework. Our proposed AMCT is extensively evaluated on seven popular benchmark datasets, and both quantitative and qualitative results firmly demonstrate its effectiveness when compared with the state-of-the-art methods.

{{</citation>}}


### (34/185) Towards Foundation Models for Learning on Tabular Data (Han Zhang et al., 2023)

{{<citation>}}

Han Zhang, Xumeng Wen, Shun Zheng, Wei Xu, Jiang Bian. (2023)  
**Towards Foundation Models for Learning on Tabular Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.07338v1)  

---


**ABSTRACT**  
Learning on tabular data underpins numerous real-world applications. Despite considerable efforts in developing effective learning models for tabular data, current transferable tabular models remain in their infancy, limited by either the lack of support for direct instruction following in new tasks or the neglect of acquiring foundational knowledge and capabilities from diverse tabular datasets. In this paper, we propose Tabular Foundation Models (TabFMs) to overcome these limitations. TabFMs harness the potential of generative tabular learning, employing a pre-trained large language model (LLM) as the base model and fine-tuning it using purpose-designed objectives on an extensive range of tabular datasets. This approach endows TabFMs with a profound understanding and universal capabilities essential for learning on tabular data. Our evaluations underscore TabFM's effectiveness: not only does it significantly excel in instruction-following tasks like zero-shot and in-context inference, but it also showcases performance that approaches, and in instances, even transcends, the renowned yet mysterious closed-source LLMs like GPT-4. Furthermore, when fine-tuning with scarce data, our model achieves remarkable efficiency and maintains competitive performance with abundant training data. Finally, while our results are promising, we also delve into TabFM's limitations and potential opportunities, aiming to stimulate and expedite future research on developing more potent TabFMs.

{{</citation>}}


### (35/185) Classification of Dysarthria based on the Levels of Severity. A Systematic Review (Afnan Al-Ali et al., 2023)

{{<citation>}}

Afnan Al-Ali, Somaya Al-Maadeed, Moutaz Saleh, Rani Chinnappa Naidu, Zachariah C Alex, Prakash Ramachandran, Rajeev Khoodeeram, Rajesh Kumar M. (2023)  
**Classification of Dysarthria based on the Levels of Severity. A Systematic Review**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07264v1)  

---


**ABSTRACT**  
Dysarthria is a neurological speech disorder that can significantly impact affected individuals' communication abilities and overall quality of life. The accurate and objective classification of dysarthria and the determination of its severity are crucial for effective therapeutic intervention. While traditional assessments by speech-language pathologists (SLPs) are common, they are often subjective, time-consuming, and can vary between practitioners. Emerging machine learning-based models have shown the potential to provide a more objective dysarthria assessment, enhancing diagnostic accuracy and reliability. This systematic review aims to comprehensively analyze current methodologies for classifying dysarthria based on severity levels. Specifically, this review will focus on determining the most effective set and type of features that can be used for automatic patient classification and evaluating the best AI techniques for this purpose. We will systematically review the literature on the automatic classification of dysarthria severity levels. Sources of information will include electronic databases and grey literature. Selection criteria will be established based on relevance to the research questions. Data extraction will include methodologies used, the type of features extracted for classification, and AI techniques employed. The findings of this systematic review will contribute to the current understanding of dysarthria classification, inform future research, and support the development of improved diagnostic tools. The implications of these findings could be significant in advancing patient care and improving therapeutic outcomes for individuals affected by dysarthria.

{{</citation>}}


### (36/185) Are GATs Out of Balance? (Nimrah Mustafa et al., 2023)

{{<citation>}}

Nimrah Mustafa, Aleksandar Bojchevski, Rebekka Burkholz. (2023)  
**Are GATs Out of Balance?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, GNN, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2310.07235v1)  

---


**ABSTRACT**  
While the expressive power and computational capabilities of graph neural networks (GNNs) have been theoretically studied, their optimization and learning dynamics, in general, remain largely unexplored. Our study undertakes the Graph Attention Network (GAT), a popular GNN architecture in which a node's neighborhood aggregation is weighted by parameterized attention coefficients. We derive a conservation law of GAT gradient flow dynamics, which explains why a high portion of parameters in GATs with standard initialization struggle to change during training. This effect is amplified in deeper GATs, which perform significantly worse than their shallow counterparts. To alleviate this problem, we devise an initialization scheme that balances the GAT network. Our approach i) allows more effective propagation of gradients and in turn enables trainability of deeper networks, and ii) attains a considerable speedup in training and convergence time in comparison to the standard initialization. Our main theorem serves as a stepping stone to studying the learning dynamics of positive homogeneous models with attention mechanisms.

{{</citation>}}


### (37/185) Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality (Liyuan Wang et al., 2023)

{{<citation>}}

Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, Jun Zhu. (2023)  
**Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.07234v1)  

---


**ABSTRACT**  
Prompt-based continual learning is an emerging direction in leveraging pre-trained knowledge for downstream continual learning, and has almost reached the performance pinnacle under supervised pre-training. However, our empirical research reveals that the current strategies fall short of their full potential under the more realistic self-supervised pre-training, which is essential for handling vast quantities of unlabeled data in practice. This is largely due to the difficulty of task-specific knowledge being incorporated into instructed representations via prompt parameters and predicted by uninstructed representations at test time. To overcome the exposed sub-optimality, we conduct a theoretical analysis of the continual learning objective in the context of pre-training, and decompose it into hierarchical components: within-task prediction, task-identity inference, and task-adaptive prediction. Following these empirical and theoretical insights, we propose Hierarchical Decomposition (HiDe-)Prompt, an innovative approach that explicitly optimizes the hierarchical components with an ensemble of task-specific prompts and statistics of both uninstructed and instructed representations, further with the coordination of a contrastive regularization strategy. Our extensive experiments demonstrate the superior performance of HiDe-Prompt and its robustness to pre-training paradigms in continual learning (e.g., up to 15.01% and 9.61% lead on Split CIFAR-100 and Split ImageNet-R, respectively). Our code is available at \url{https://github.com/thu-ml/HiDe-Prompt}.

{{</citation>}}


### (38/185) Improved Membership Inference Attacks Against Language Classification Models (Shlomit Shachor et al., 2023)

{{<citation>}}

Shlomit Shachor, Natalia Razinkov, Abigail Goldsteen. (2023)  
**Improved Membership Inference Attacks Against Language Classification Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07219v1)  

---


**ABSTRACT**  
Artificial intelligence systems are prevalent in everyday life, with use cases in retail, manufacturing, health, and many other fields. With the rise in AI adoption, associated risks have been identified, including privacy risks to the people whose data was used to train models. Assessing the privacy risks of machine learning models is crucial to enabling knowledgeable decisions on whether to use, deploy, or share a model. A common approach to privacy risk assessment is to run one or more known attacks against the model and measure their success rate. We present a novel framework for running membership inference attacks against classification models. Our framework takes advantage of the ensemble method, generating many specialized attack models for different subsets of the data. We show that this approach achieves higher accuracy than either a single attack model or an attack model per class label, both on classical and language classification tasks.

{{</citation>}}


### (39/185) Enhancing Neural Architecture Search with Multiple Hardware Constraints for Deep Learning Model Deployment on Tiny IoT Devices (Alessio Burrello et al., 2023)

{{<citation>}}

Alessio Burrello, Matteo Risso, Beatrice Alessandra Motetti, Enrico Macii, Luca Benini, Daniele Jahier Pagliari. (2023)  
**Enhancing Neural Architecture Search with Multiple Hardware Constraints for Deep Learning Model Deployment on Tiny IoT Devices**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.07217v1)  

---


**ABSTRACT**  
The rapid proliferation of computing domains relying on Internet of Things (IoT) devices has created a pressing need for efficient and accurate deep-learning (DL) models that can run on low-power devices. However, traditional DL models tend to be too complex and computationally intensive for typical IoT end-nodes. To address this challenge, Neural Architecture Search (NAS) has emerged as a popular design automation technique for co-optimizing the accuracy and complexity of deep neural networks. Nevertheless, existing NAS techniques require many iterations to produce a network that adheres to specific hardware constraints, such as the maximum memory available on the hardware or the maximum latency allowed by the target application. In this work, we propose a novel approach to incorporate multiple constraints into so-called Differentiable NAS optimization methods, which allows the generation, in a single shot, of a model that respects user-defined constraints on both memory and latency in a time comparable to a single standard training. The proposed approach is evaluated on five IoT-relevant benchmarks, including the MLPerf Tiny suite and Tiny ImageNet, demonstrating that, with a single search, it is possible to reduce memory and latency by 87.4% and 54.2%, respectively (as defined by our targets), while ensuring non-inferior accuracy on state-of-the-art hand-tuned deep neural networks for TinyML.

{{</citation>}}


### (40/185) Robust Safe Reinforcement Learning under Adversarial Disturbances (Zeyang Li et al., 2023)

{{<citation>}}

Zeyang Li, Chuxiong Hu, Shengbo Eben Li, Jia Cheng, Yunan Wang. (2023)  
**Robust Safe Reinforcement Learning under Adversarial Disturbances**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07207v1)  

---


**ABSTRACT**  
Safety is a primary concern when applying reinforcement learning to real-world control tasks, especially in the presence of external disturbances. However, existing safe reinforcement learning algorithms rarely account for external disturbances, limiting their applicability and robustness in practice. To address this challenge, this paper proposes a robust safe reinforcement learning framework that tackles worst-case disturbances. First, this paper presents a policy iteration scheme to solve for the robust invariant set, i.e., a subset of the safe set, where persistent safety is only possible for states within. The key idea is to establish a two-player zero-sum game by leveraging the safety value function in Hamilton-Jacobi reachability analysis, in which the protagonist (i.e., control inputs) aims to maintain safety and the adversary (i.e., external disturbances) tries to break down safety. This paper proves that the proposed policy iteration algorithm converges monotonically to the maximal robust invariant set. Second, this paper integrates the proposed policy iteration scheme into a constrained reinforcement learning algorithm that simultaneously synthesizes the robust invariant set and uses it for constrained policy optimization. This algorithm tackles both optimality and safety, i.e., learning a policy that attains high rewards while maintaining safety under worst-case disturbances. Experiments on classic control tasks show that the proposed method achieves zero constraint violation with learned worst-case adversarial disturbances, while other baseline algorithms violate the safety constraints substantially. Our proposed method also attains comparable performance as the baselines even in the absence of the adversary.

{{</citation>}}


### (41/185) Generalized Neural Sorting Networks with Error-Free Differentiable Swap Functions (Jungtaek Kim et al., 2023)

{{<citation>}}

Jungtaek Kim, Jeongbeen Yoon, Minsu Cho. (2023)  
**Generalized Neural Sorting Networks with Error-Free Differentiable Swap Functions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07174v1)  

---


**ABSTRACT**  
Sorting is a fundamental operation of all computer systems, having been a long-standing significant research topic. Beyond the problem formulation of traditional sorting algorithms, we consider sorting problems for more abstract yet expressive inputs, e.g., multi-digit images and image fragments, through a neural sorting network. To learn a mapping from a high-dimensional input to an ordinal variable, the differentiability of sorting networks needs to be guaranteed. In this paper we define a softening error by a differentiable swap function, and develop an error-free swap function that holds non-decreasing and differentiability conditions. Furthermore, a permutation-equivariant Transformer network with multi-head attention is adopted to capture dependency between given inputs and also leverage its model capacity with self-attention. Experiments on diverse sorting benchmarks show that our methods perform better than or comparable to baseline methods.

{{</citation>}}


## cs.CL (46)



### (42/185) Crosslingual Structural Priming and the Pre-Training Dynamics of Bilingual Language Models (Catherine Arnett et al., 2023)

{{<citation>}}

Catherine Arnett, Tyler A. Chang, James A. Michaelov, Benjamin K. Bergen. (2023)  
**Crosslingual Structural Priming and the Pre-Training Dynamics of Bilingual Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07929v1)  

---


**ABSTRACT**  
Do multilingual language models share abstract grammatical representations across languages, and if so, when do these develop? Following Sinclair et al. (2022), we use structural priming to test for abstract grammatical representations with causal effects on model outputs. We extend the approach to a Dutch-English bilingual setting, and we evaluate a Dutch-English language model during pre-training. We find that crosslingual structural priming effects emerge early after exposure to the second language, with less than 1M tokens of data in that language. We discuss implications for data contamination, low-resource transfer, and how abstract grammatical representations emerge in multilingual models.

{{</citation>}}


### (43/185) Pit One Against Many: Leveraging Attention-head Embeddings for Parameter-efficient Multi-head Attention (Huiyin Xue et al., 2023)

{{<citation>}}

Huiyin Xue, Nikolaos Aletras. (2023)  
**Pit One Against Many: Leveraging Attention-head Embeddings for Parameter-efficient Multi-head Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2310.07911v1)  

---


**ABSTRACT**  
Scaling pre-trained language models has resulted in large performance gains in various natural language processing tasks but comes with a large cost in memory requirements. Inspired by the position embeddings in transformers, we aim to simplify and reduce the memory footprint of the multi-head attention (MHA) mechanism. We propose an alternative module that uses only a single shared projection matrix and multiple head embeddings (MHE), i.e. one per head. We empirically demonstrate that our MHE attention is substantially more memory efficient compared to alternative attention mechanisms while achieving high predictive performance retention ratio to vanilla MHA on several downstream tasks. MHE attention only requires a negligible fraction of additional parameters ($3nd$, where $n$ is the number of attention heads and $d$ the size of the head embeddings) compared to a single-head attention, while MHA requires $(3n^2-3n)d^2-3nd$ additional parameters.

{{</citation>}}


### (44/185) TabLib: A Dataset of 627M Tables with Context (Gus Eggert et al., 2023)

{{<citation>}}

Gus Eggert, Kevin Huo, Mike Biven, Justin Waugh. (2023)  
**TabLib: A Dataset of 627M Tables with Context**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DB, cs-LG, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07875v1)  

---


**ABSTRACT**  
It is well-established that large, diverse datasets play a pivotal role in the performance of modern AI systems for text and image modalities. However, there are no datasets for tabular data of comparable size and diversity to those available for text and images. Thus we present "TabLib'', a compilation of 627 million tables totaling 69 TiB, along with 867B tokens of context. TabLib was extracted from numerous file formats, including CSV, HTML, SQLite, PDF, Excel, and others, sourced from GitHub and Common Crawl. The size and diversity of TabLib offer considerable promise in the table modality, reminiscent of the original promise of foundational datasets for text and images, such as The Pile and LAION.

{{</citation>}}


### (45/185) Assessing Evaluation Metrics for Neural Test Oracle Generation (Jiho Shin et al., 2023)

{{<citation>}}

Jiho Shin, Hadi Hemmati, Moshi Wei, Song Wang. (2023)  
**Assessing Evaluation Metrics for Neural Test Oracle Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.07856v1)  

---


**ABSTRACT**  
In this work, we revisit existing oracle generation studies plus ChatGPT to empirically investigate the current standing of their performance in both NLG-based and test adequacy metrics. Specifically, we train and run four state-of-the-art test oracle generation models on five NLG-based and two test adequacy metrics for our analysis. We apply two different correlation analyses between these two different sets of metrics. Surprisingly, we found no significant correlation between the NLG-based metrics and test adequacy metrics. For instance, oracles generated from ChatGPT on the project activemq-artemis had the highest performance on all the NLG-based metrics among the studied NOGs, however, it had the most number of projects with a decrease in test adequacy metrics compared to all the studied NOGs. We further conduct a qualitative analysis to explore the reasons behind our observations, we found that oracles with high NLG-based metrics but low test adequacy metrics tend to have complex or multiple chained method invocations within the oracle's parameters, making it hard for the model to generate completely, affecting the test adequacy metrics. On the other hand, oracles with low NLG-based metrics but high test adequacy metrics tend to have to call different assertion types or a different method that functions similarly to the ones in the ground truth. Overall, this work complements prior studies on test oracle generation with an extensive performance evaluation with both NLG and test adequacy metrics and provides guidelines for better assessment of deep learning applications in software test generation in the future.

{{</citation>}}


### (46/185) Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations (Zhuoyan Li et al., 2023)

{{<citation>}}

Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, Ming Yin. (2023)  
**Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Text Classification  
[Paper Link](http://arxiv.org/abs/2310.07849v2)  

---


**ABSTRACT**  
The collection and curation of high-quality training data is crucial for developing text classification models with superior performance, but it is often associated with significant costs and time investment. Researchers have recently explored using large language models (LLMs) to generate synthetic datasets as an alternative approach. However, the effectiveness of the LLM-generated synthetic data in supporting model training is inconsistent across different classification tasks. To better understand factors that moderate the effectiveness of the LLM-generated synthetic data, in this study, we look into how the performance of models trained on these synthetic data may vary with the subjectivity of classification. Our results indicate that subjectivity, at both the task level and instance level, is negatively associated with the performance of the model trained on synthetic data. We conclude by discussing the implications of our work on the potential and limitations of leveraging LLM for synthetic data generation.

{{</citation>}}


### (47/185) Framework for Question-Answering in Sanskrit through Automated Construction of Knowledge Graphs (Hrishikesh Terdalkar et al., 2023)

{{<citation>}}

Hrishikesh Terdalkar, Arnab Bhattacharya. (2023)  
**Framework for Question-Answering in Sanskrit through Automated Construction of Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.07848v1)  

---


**ABSTRACT**  
Sanskrit (sa\d{m}sk\d{r}ta) enjoys one of the largest and most varied literature in the whole world. Extracting the knowledge from it, however, is a challenging task due to multiple reasons including complexity of the language and paucity of standard natural language processing tools. In this paper, we target the problem of building knowledge graphs for particular types of relationships from sa\d{m}sk\d{r}ta texts. We build a natural language question-answering system in sa\d{m}sk\d{r}ta that uses the knowledge graph to answer factoid questions. We design a framework for the overall system and implement two separate instances of the system on human relationships from mah\=abh\=arata and r\=am\=aya\d{n}a, and one instance on synonymous relationships from bh\=avaprak\=a\'sa nigha\d{n}\d{t}u, a technical text from \=ayurveda. We show that about 50% of the factoid questions can be answered correctly by the system. More importantly, we analyse the shortcomings of the system in detail for each step, and discuss the possible ways forward.

{{</citation>}}


### (48/185) Does Synthetic Data Make Large Language Models More Efficient? (Sia Gholami et al., 2023)

{{<citation>}}

Sia Gholami, Marwan Omar. (2023)  
**Does Synthetic Data Make Large Language Models More Efficient?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.07830v1)  

---


**ABSTRACT**  
Natural Language Processing (NLP) has undergone transformative changes with the advent of deep learning methodologies. One challenge persistently confronting researchers is the scarcity of high-quality, annotated datasets that drive these models. This paper explores the nuances of synthetic data generation in NLP, with a focal point on template-based question generation. By assessing its advantages, including data augmentation potential and the introduction of structured variety, we juxtapose these benefits against inherent limitations, such as the risk of overfitting and the constraints posed by pre-defined templates. Drawing from empirical evaluations, we demonstrate the impact of template-based synthetic data on the performance of modern transformer models. We conclude by emphasizing the delicate balance required between synthetic and real-world data, and the future trajectories of integrating synthetic data in model training pipelines. The findings aim to guide NLP practitioners in harnessing synthetic data's potential, ensuring optimal model performance in diverse applications.

{{</citation>}}


### (49/185) Antarlekhaka: A Comprehensive Tool for Multi-task Natural Language Annotation (Hrishikesh Terdalkar et al., 2023)

{{<citation>}}

Hrishikesh Terdalkar, Arnab Bhattacharya. (2023)  
**Antarlekhaka: A Comprehensive Tool for Multi-task Natural Language Annotation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.07826v1)  

---


**ABSTRACT**  
One of the primary obstacles in the advancement of Natural Language Processing (NLP) technologies for low-resource languages is the lack of annotated datasets for training and testing machine learning models. In this paper, we present Antarlekhaka, a tool for manual annotation of a comprehensive set of tasks relevant to NLP. The tool is Unicode-compatible, language-agnostic, Web-deployable and supports distributed annotation by multiple simultaneous annotators. The system sports user-friendly interfaces for 8 categories of annotation tasks. These, in turn, enable the annotation of a considerably larger set of NLP tasks. The task categories include two linguistic tasks not handled by any other tool, namely, sentence boundary detection and deciding canonical word order, which are important tasks for text that is in the form of poetry. We propose the idea of sequential annotation based on small text units, where an annotator performs several tasks related to a single text unit before proceeding to the next unit. The research applications of the proposed mode of multi-task annotation are also discussed. Antarlekhaka outperforms other annotation tools in objective evaluation. It has been also used for two real-life annotation tasks on two different languages, namely, Sanskrit and Bengali. The tool is available at https://github.com/Antarlekhaka/code.

{{</citation>}}


### (50/185) Non-autoregressive Text Editing with Copy-aware Latent Alignments (Yu Zhang et al., 2023)

{{<citation>}}

Yu Zhang, Yue Zhang, Leyang Cui, Guohong Fu. (2023)  
**Non-autoregressive Text Editing with Copy-aware Latent Alignments**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Seq2Seq  
[Paper Link](http://arxiv.org/abs/2310.07821v1)  

---


**ABSTRACT**  
Recent work has witnessed a paradigm shift from Seq2Seq to Seq2Edit in the field of text editing, with the aim of addressing the slow autoregressive inference problem posed by the former. Despite promising results, Seq2Edit approaches still face several challenges such as inflexibility in generation and difficulty in generalizing to other languages. In this work, we propose a novel non-autoregressive text editing method to circumvent the above issues, by modeling the edit process with latent CTC alignments. We make a crucial extension to CTC by introducing the copy operation into the edit space, thus enabling more efficient management of textual overlap in editing. We conduct extensive experiments on GEC and sentence fusion tasks, showing that our proposed method significantly outperforms existing Seq2Edit models and achieves similar or even better results than Seq2Seq with over $4\times$ speedup. Moreover, it demonstrates good generalizability on German and Russian. In-depth analyses reveal the strengths of our method in terms of the robustness under various scenarios and generating fluent and flexible outputs.

{{</citation>}}


### (51/185) Faithfulness Measurable Masked Language Models (Andreas Madsen et al., 2023)

{{<citation>}}

Andreas Madsen, Siva Reddy, Sarath Chandar. (2023)  
**Faithfulness Measurable Masked Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.07819v1)  

---


**ABSTRACT**  
A common approach to explain NLP models, is to use importance measures that express which tokens are important for a prediction. Unfortunately, such explanations are often wrong despite being persuasive. Therefore, it is essential to measure their faithfulness. One such metric is if tokens are truly important, then masking them should result in worse model performance. However, token masking introduces out-of-distribution issues and existing solutions are computationally expensive and employ proxy-models. Furthermore, other metrics are very limited in scope. In this work, we propose an inherently faithfulness measurable model that addresses these challenges. This is achieved by using a novel fine-tuning method that incorporates masking, such that masking tokens become in-distribution by design. This differs from existing approaches, which are completely model-agnostic but are inapplicable in practice. We demonstrate the generality of our approach by applying it to various tasks and validate it using statistical in-distribution tests. Additionally, because masking is in-distribution, importance measures which themselves use masking become more faithful, thus our model becomes more explainable.

{{</citation>}}


### (52/185) Exploring the Relationship between Analogy Identification and Sentence Structure Encoding in Large Language Models (Thilini Wijesiriwardene et al., 2023)

{{<citation>}}

Thilini Wijesiriwardene, Ruwan Wickramarachchi, Aishwarya Naresh Reganti, Vinija Jain, Aman Chadha, Amit Sheth, Amitava Das. (2023)  
**Exploring the Relationship between Analogy Identification and Sentence Structure Encoding in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.07818v1)  

---


**ABSTRACT**  
Identifying analogies plays a pivotal role in human cognition and language proficiency. In the last decade, there has been extensive research on word analogies in the form of ``A is to B as C is to D.'' However, there is a growing interest in analogies that involve longer text, such as sentences and collections of sentences, which convey analogous meanings. While the current NLP research community evaluates the ability of Large Language Models (LLMs) to identify such analogies, the underlying reasons behind these abilities warrant deeper investigation. Furthermore, the capability of LLMs to encode both syntactic and semantic structures of language within their embeddings has garnered significant attention with the surge in their utilization. In this work, we examine the relationship between the abilities of multiple LLMs to identify sentence analogies, and their capacity to encode syntactic and semantic structures. Through our analysis, we find that analogy identification ability of LLMs is positively correlated with their ability to encode syntactic and semantic structures of sentences. Specifically, we find that the LLMs which capture syntactic structures better, also have higher abilities in identifying sentence analogies.

{{</citation>}}


### (53/185) Ontology Enrichment for Effective Fine-grained Entity Typing (Siru Ouyang et al., 2023)

{{<citation>}}

Siru Ouyang, Jiaxin Huang, Pranav Pillai, Yunyi Zhang, Yu Zhang, Jiawei Han. (2023)  
**Ontology Enrichment for Effective Fine-grained Entity Typing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.07795v1)  

---


**ABSTRACT**  
Fine-grained entity typing (FET) is the task of identifying specific entity types at a fine-grained level for entity mentions based on their contextual information. Conventional methods for FET require extensive human annotation, which is time-consuming and costly. Recent studies have been developing weakly supervised or zero-shot approaches. We study the setting of zero-shot FET where only an ontology is provided. However, most existing ontology structures lack rich supporting information and even contain ambiguous relations, making them ineffective in guiding FET. Recently developed language models, though promising in various few-shot and zero-shot NLP tasks, may face challenges in zero-shot FET due to their lack of interaction with task-specific ontology. In this study, we propose OnEFET, where we (1) enrich each node in the ontology structure with two types of extra information: instance information for training sample augmentation and topic information to relate types to contexts, and (2) develop a coarse-to-fine typing algorithm that exploits the enriched information by training an entailment model with contrasting topics and instance-based augmented training samples. Our experiments show that OnEFET achieves high-quality fine-grained entity typing without human annotation, outperforming existing zero-shot methods by a large margin and rivaling supervised methods.

{{</citation>}}


### (54/185) GenTKG: Generative Forecasting on Temporal Knowledge Graph (Ruotong Liao et al., 2023)

{{<citation>}}

Ruotong Liao, Xu Jia, Yunpu Ma, Volker Tresp. (2023)  
**GenTKG: Generative Forecasting on Temporal Knowledge Graph**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.07793v1)  

---


**ABSTRACT**  
The rapid advancements in large language models (LLMs) have ignited interest in the temporal knowledge graph (tKG) domain, where conventional carefully designed embedding-based and rule-based models dominate. The question remains open of whether pre-trained LLMs can understand structured temporal relational data and replace them as the foundation model for temporal relational forecasting. Therefore, we bring temporal knowledge forecasting into the generative setting. However, challenges occur in the huge chasms between complex temporal graph data structure and sequential natural expressions LLMs can handle, and between the enormous data sizes of tKGs and heavy computation costs of finetuning LLMs. To address these challenges, we propose a novel retrieval augmented generation framework that performs generative forecasting on tKGs named GenTKG, which combines a temporal logical rule-based retrieval strategy and lightweight parameter-efficient instruction tuning. Extensive experiments have shown that GenTKG outperforms conventional methods of temporal relational forecasting under low computation resources. GenTKG also highlights remarkable transferability with exceeding performance on unseen datasets without re-training. Our work reveals the huge potential of LLMs in the tKG domain and opens a new frontier for generative forecasting on tKGs.

{{</citation>}}


### (55/185) To Build Our Future, We Must Know Our Past: Contextualizing Paradigm Shifts in Natural Language Processing (Sireesh Gururaja et al., 2023)

{{<citation>}}

Sireesh Gururaja, Amanda Bertsch, Clara Na, David Gray Widder, Emma Strubell. (2023)  
**To Build Our Future, We Must Know Our Past: Contextualizing Paradigm Shifts in Natural Language Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.07715v1)  

---


**ABSTRACT**  
NLP is in a period of disruptive change that is impacting our methodologies, funding sources, and public perception. In this work, we seek to understand how to shape our future by better understanding our past. We study factors that shape NLP as a field, including culture, incentives, and infrastructure by conducting long-form interviews with 26 NLP researchers of varying seniority, research area, institution, and social identity. Our interviewees identify cyclical patterns in the field, as well as new shifts without historical parallel, including changes in benchmark culture and software infrastructure. We complement this discussion with quantitative analysis of citation, authorship, and language use in the ACL Anthology over time. We conclude by discussing shared visions, concerns, and hopes for the future of NLP. We hope that this study of our field's past and present can prompt informed discussion of our community's implicit norms and more deliberate action to consciously shape the future.

{{</citation>}}


### (56/185) InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining (Boxin Wang et al., 2023)

{{<citation>}}

Boxin Wang, Wei Ping, Lawrence McAfee, Peng Xu, Bo Li, Mohammad Shoeybi, Bryan Catanzaro. (2023)  
**InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: GPT, QA  
[Paper Link](http://arxiv.org/abs/2310.07713v1)  

---


**ABSTRACT**  
Pretraining auto-regressive large language models (LLMs) with retrieval demonstrates better perplexity and factual accuracy by leveraging external databases. However, the size of existing pretrained retrieval-augmented LLM is still limited (e.g., Retro has 7.5B parameters), which limits the effectiveness of instruction tuning and zero-shot generalization. In this work, we introduce Retro 48B, the largest LLM pretrained with retrieval before instruction tuning. Specifically, we continue to pretrain the 43B GPT model on additional 100 billion tokens using the Retro augmentation method by retrieving from 1.2 trillion tokens. The obtained foundation model, Retro 48B, largely outperforms the original 43B GPT in terms of perplexity. After instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on zero-shot question answering (QA) tasks. Specifically, the average improvement of InstructRetro is 7% over its GPT counterpart across 8 short-form QA tasks, and 10% over GPT across 4 challenging long-form QA tasks. Surprisingly, we find that one can ablate the encoder from InstructRetro architecture and directly use its decoder backbone, while achieving comparable results. We hypothesize that pretraining with retrieval makes its decoder good at incorporating context for QA. Our results highlights the promising direction to obtain a better GPT decoder for QA through continued pretraining with retrieval before instruction tuning.

{{</citation>}}


### (57/185) Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models (Raphael Tang et al., 2023)

{{<citation>}}

Raphael Tang, Xinyu Zhang, Xueguang Ma, Jimmy Lin, Ferhan Ture. (2023)  
**Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07712v1)  

---


**ABSTRACT**  
Large language models (LLMs) exhibit positional bias in how they use context, which especially complicates listwise ranking. To address this, we propose permutation self-consistency, a form of self-consistency over ranking list outputs of black-box LLMs. Our key idea is to marginalize out different list orders in the prompt to produce an order-independent ranking with less positional bias. First, given some input prompt, we repeatedly shuffle the list in the prompt and pass it through the LLM while holding the instructions the same. Next, we aggregate the resulting sample of rankings by computing the central ranking closest in distance to all of them, marginalizing out prompt order biases in the process. Theoretically, we prove the robustness of our method, showing convergence to the true ranking in the presence of random perturbations. Empirically, on five list-ranking datasets in sorting and passage reranking, our approach improves scores from conventional inference by up to 7-18% for GPT-3.5 and 8-16% for LLaMA v2 (70B), surpassing the previous state of the art in passage reranking. Our code is at https://github.com/castorini/perm-sc.

{{</citation>}}


### (58/185) Well Begun is Half Done: Generator-agnostic Knowledge Pre-Selection for Knowledge-Grounded Dialogue (Lang Qin et al., 2023)

{{<citation>}}

Lang Qin, Yao Zhang, Hongru Liang, Jun Wang, Zhenglu Yang. (2023)  
**Well Begun is Half Done: Generator-agnostic Knowledge Pre-Selection for Knowledge-Grounded Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2310.07659v2)  

---


**ABSTRACT**  
Accurate knowledge selection is critical in knowledge-grounded dialogue systems. Towards a closer look at it, we offer a novel perspective to organize existing literature, i.e., knowledge selection coupled with, after, and before generation. We focus on the third under-explored category of study, which can not only select knowledge accurately in advance, but has the advantage to reduce the learning, adjustment, and interpretation burden of subsequent response generation models, especially LLMs. We propose GATE, a generator-agnostic knowledge selection method, to prepare knowledge for subsequent response generation models by selecting context-related knowledge among different knowledge structures and variable knowledge requirements. Experimental results demonstrate the superiority of GATE, and indicate that knowledge selection before generation is a lightweight yet effective way to facilitate LLMs (e.g., ChatGPT) to generate more informative responses.

{{</citation>}}


### (59/185) Evaluating Large Language Models at Evaluating Instruction Following (Zhiyuan Zeng et al., 2023)

{{<citation>}}

Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, Danqi Chen. (2023)  
**Evaluating Large Language Models at Evaluating Instruction Following**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07641v1)  

---


**ABSTRACT**  
As research in large language models (LLMs) continues to accelerate, LLM-based evaluation has emerged as a scalable and cost-effective alternative to human evaluations for comparing the ever increasing list of models. This paper investigates the efficacy of these "LLM evaluators", particularly in using them to assess instruction following, a metric that gauges how closely generated text adheres to the given instruction. We introduce a challenging meta-evaluation benchmark, LLMBar, designed to test the ability of an LLM evaluator in discerning instruction-following outputs. The authors manually curated 419 pairs of outputs, one adhering to instructions while the other diverging, yet may possess deceptive qualities that mislead an LLM evaluator, e.g., a more engaging tone. Contrary to existing meta-evaluation, we discover that different evaluators (i.e., combinations of LLMs and prompts) exhibit distinct performance on LLMBar and even the highest-scoring ones have substantial room for improvement. We also present a novel suite of prompting strategies that further close the gap between LLM and human evaluators. With LLMBar, we hope to offer more insight into LLM evaluators and foster future research in developing better instruction-following models.

{{</citation>}}


### (60/185) The Past, Present and Better Future of Feedback Learning in Large Language Models for Subjective Human Preferences and Values (Hannah Rose Kirk et al., 2023)

{{<citation>}}

Hannah Rose Kirk, Andrew M. Bean, Bertie Vidgen, Paul Röttger, Scott A. Hale. (2023)  
**The Past, Present and Better Future of Feedback Learning in Large Language Models for Subjective Human Preferences and Values**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07629v1)  

---


**ABSTRACT**  
Human feedback is increasingly used to steer the behaviours of Large Language Models (LLMs). However, it is unclear how to collect and incorporate feedback in a way that is efficient, effective and unbiased, especially for highly subjective human preferences and values. In this paper, we survey existing approaches for learning from human feedback, drawing on 95 papers primarily from the ACL and arXiv repositories.First, we summarise the past, pre-LLM trends for integrating human feedback into language models. Second, we give an overview of present techniques and practices, as well as the motivations for using feedback; conceptual frameworks for defining values and preferences; and how feedback is collected and from whom. Finally, we encourage a better future of feedback learning in LLMs by raising five unresolved conceptual and practical challenges.

{{</citation>}}


### (61/185) Democratizing LLMs: An Exploration of Cost-Performance Trade-offs in Self-Refined Open-Source Models (Sumuk Shashidhar et al., 2023)

{{<citation>}}

Sumuk Shashidhar, Abhinav Chinta, Vaibhav Sahai, Zhenhailong Wang, Heng Ji. (2023)  
**Democratizing LLMs: An Exploration of Cost-Performance Trade-offs in Self-Refined Open-Source Models**  

---
Primary Category: cs.CL  
Categories: 68T50 (Primary), I-2-7; A-2; H-3-4; K-4-1; C-4, cs-AI, cs-CL, cs-PF, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.07611v1)  

---


**ABSTRACT**  
The dominance of proprietary LLMs has led to restricted access and raised information privacy concerns. High-performing open-source alternatives are crucial for information-sensitive and high-volume applications but often lag behind in performance. To address this gap, we propose (1) A untargeted variant of iterative self-critique and self-refinement devoid of external influence. (2) A novel ranking metric - Performance, Refinement, and Inference Cost Score (PeRFICS) - to find the optimal model for a given task considering refined performance and cost. Our experiments show that SoTA open source models of varying sizes from 7B - 65B, on average, improve 8.2% from their baseline performance. Strikingly, even models with extremely small memory footprints, such as Vicuna-7B, show a 11.74% improvement overall and up to a 25.39% improvement in high-creativity, open ended tasks on the Vicuna benchmark. Vicuna-13B takes it a step further and outperforms ChatGPT post-refinement. This work has profound implications for resource-constrained and information-sensitive environments seeking to leverage LLMs without incurring prohibitive costs, compromising on performance and privacy. The domain-agnostic self-refinement process coupled with our novel ranking metric facilitates informed decision-making in model selection, thereby reducing costs and democratizing access to high-performing language models, as evidenced by case studies.

{{</citation>}}


### (62/185) QACHECK: A Demonstration System for Question-Guided Multi-Hop Fact-Checking (Liangming Pan et al., 2023)

{{<citation>}}

Liangming Pan, Xinyuan Lu, Min-Yen Kan, Preslav Nakov. (2023)  
**QACHECK: A Demonstration System for Question-Guided Multi-Hop Fact-Checking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fact-Checking, QA  
[Paper Link](http://arxiv.org/abs/2310.07609v1)  

---


**ABSTRACT**  
Fact-checking real-world claims often requires complex, multi-step reasoning due to the absence of direct evidence to support or refute them. However, existing fact-checking systems often lack transparency in their decision-making, making it challenging for users to comprehend their reasoning process. To address this, we propose the Question-guided Multi-hop Fact-Checking (QACHECK) system, which guides the model's reasoning process by asking a series of questions critical for verifying a claim. QACHECK has five key modules: a claim verifier, a question generator, a question-answering module, a QA validator, and a reasoner. Users can input a claim into QACHECK, which then predicts its veracity and provides a comprehensive report detailing its reasoning process, guided by a sequence of (question, answer) pairs. QACHECK also provides the source of evidence supporting each question, fostering a transparent, explainable, and user-friendly fact-checking process. A recorded video of QACHECK is at https://www.youtube.com/watch?v=ju8kxSldM64

{{</citation>}}


### (63/185) Accurate Use of Label Dependency in Multi-Label Text Classification Through the Lens of Causality (Caoyun Fan et al., 2023)

{{<citation>}}

Caoyun Fan, Wenqing Chen, Jidong Tian, Yitian Li, Hao He, Yaohui Jin. (2023)  
**Accurate Use of Label Dependency in Multi-Label Text Classification Through the Lens of Causality**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2310.07588v1)  

---


**ABSTRACT**  
Multi-Label Text Classification (MLTC) aims to assign the most relevant labels to each given text. Existing methods demonstrate that label dependency can help to improve the model's performance. However, the introduction of label dependency may cause the model to suffer from unwanted prediction bias. In this study, we attribute the bias to the model's misuse of label dependency, i.e., the model tends to utilize the correlation shortcut in label dependency rather than fusing text information and label dependency for prediction. Motivated by causal inference, we propose a CounterFactual Text Classifier (CFTC) to eliminate the correlation bias, and make causality-based predictions. Specifically, our CFTC first adopts the predict-then-modify backbone to extract precise label information embedded in label dependency, then blocks the correlation shortcut through the counterfactual de-bias technique with the help of the human causal graph. Experimental results on three datasets demonstrate that our CFTC significantly outperforms the baselines and effectively eliminates the correlation bias in datasets.

{{</citation>}}


### (64/185) Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity (Cunxiang Wang et al., 2023)

{{<citation>}}

Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, Linyi Yang, Jindong Wang, Xing Xie, Zheng Zhang, Yue Zhang. (2023)  
**Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07521v1)  

---


**ABSTRACT**  
This survey addresses the crucial issue of factuality in Large Language Models (LLMs). As LLMs find applications across diverse domains, the reliability and accuracy of their outputs become vital. We define the Factuality Issue as the probability of LLMs to produce content inconsistent with established facts. We first delve into the implications of these inaccuracies, highlighting the potential consequences and challenges posed by factual errors in LLM outputs. Subsequently, we analyze the mechanisms through which LLMs store and process facts, seeking the primary causes of factual errors. Our discussion then transitions to methodologies for evaluating LLM factuality, emphasizing key metrics, benchmarks, and studies. We further explore strategies for enhancing LLM factuality, including approaches tailored for specific domains. We focus two primary LLM configurations standalone LLMs and Retrieval-Augmented LLMs that utilizes external data, we detail their unique challenges and potential enhancements. Our survey offers a structured guide for researchers aiming to fortify the factual reliability of LLMs.

{{</citation>}}


### (65/185) KwaiYiiMath: Technical Report (Jiayi Fu et al., 2023)

{{<citation>}}

Jiayi Fu, Lei Lin, Xiaoyang Gao, Pengli Liu, Zhengzong Chen, Zhirui Yang, Shengnan Zhang, Xue Zheng, Yan Li, Yuliang Liu, Xucheng Ye, Yiqiao Liao, Chao Liao, Bin Chen, Chengru Song, Junchen Wan, Zijia Lin, Fuzheng Zhang, Zhongyuan Wang, Di Zhang, Kun Gai. (2023)  
**KwaiYiiMath: Technical Report**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.07488v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) have demonstrated remarkable abilities in handling a variety of natural language processing (NLP) downstream tasks, even on mathematical tasks requiring multi-step reasoning. In this report, we introduce the KwaiYiiMath which enhances the mathematical reasoning abilities of KwaiYiiBase1, by applying Supervised Fine-Tuning (SFT) and Reinforced Learning from Human Feedback (RLHF), including on both English and Chinese mathematical tasks. Meanwhile, we also constructed a small-scale Chinese primary school mathematics test set (named KMath), consisting of 188 examples to evaluate the correctness of the problem-solving process generated by the models. Empirical studies demonstrate that KwaiYiiMath can achieve state-of-the-art (SOTA) performance on GSM8k, CMath, and KMath compared with the similar size models, respectively.

{{</citation>}}


### (66/185) Cognate Transformer for Automated Phonological Reconstruction and Cognate Reflex Prediction (V. S. D. S. Mahesh Akavarapu et al., 2023)

{{<citation>}}

V. S. D. S. Mahesh Akavarapu, Arnab Bhattacharya. (2023)  
**Cognate Transformer for Automated Phonological Reconstruction and Cognate Reflex Prediction**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07487v1)  

---


**ABSTRACT**  
Phonological reconstruction is one of the central problems in historical linguistics where a proto-word of an ancestral language is determined from the observed cognate words of daughter languages. Computational approaches to historical linguistics attempt to automate the task by learning models on available linguistic data. Several ideas and techniques drawn from computational biology have been successfully applied in the area of computational historical linguistics. Following these lines, we adapt MSA Transformer, a protein language model, to the problem of automated phonological reconstruction. MSA Transformer trains on multiple sequence alignments as input and is, thus, apt for application on aligned cognate words. We, hence, name our model as Cognate Transformer. We also apply the model on another associated task, namely, cognate reflex prediction, where a reflex word in a daughter language is predicted based on cognate words from other daughter languages. We show that our model outperforms the existing models on both tasks, especially when it is pre-trained on masked word prediction task.

{{</citation>}}


### (67/185) Adapting the adapters for code-switching in multilingual ASR (Atharva Kulkarni et al., 2023)

{{<citation>}}

Atharva Kulkarni, Ajinkya Kulkarni, Miguel Couceiro, Hanan Aldarmaki. (2023)  
**Adapting the adapters for code-switching in multilingual ASR**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.07423v1)  

---


**ABSTRACT**  
Recently, large pre-trained multilingual speech models have shown potential in scaling Automatic Speech Recognition (ASR) to many low-resource languages. Some of these models employ language adapters in their formulation, which helps to improve monolingual performance and avoids some of the drawbacks of multi-lingual modeling on resource-rich languages. However, this formulation restricts the usability of these models on code-switched speech, where two languages are mixed together in the same utterance. In this work, we propose ways to effectively fine-tune such models on code-switched speech, by assimilating information from both language adapters at each language adaptation point in the network. We also model code-switching as a sequence of latent binary sequences that can be used to guide the flow of information from each language adapter at the frame level. The proposed approaches are evaluated on three code-switched datasets encompassing Arabic, Mandarin, and Hindi languages paired with English, showing consistent improvements in code-switching performance with at least 10\% absolute reduction in CER across all test sets.

{{</citation>}}


### (68/185) DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation (Qingkai Fang et al., 2023)

{{<citation>}}

Qingkai Fang, Yan Zhou, Yang Feng. (2023)  
**DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07403v1)  

---


**ABSTRACT**  
Direct speech-to-speech translation (S2ST) translates speech from one language into another using a single model. However, due to the presence of linguistic and acoustic diversity, the target speech follows a complex multimodal distribution, posing challenges to achieving both high-quality translations and fast decoding speeds for S2ST models. In this paper, we propose DASpeech, a non-autoregressive direct S2ST model which realizes both fast and high-quality S2ST. To better capture the complex distribution of the target speech, DASpeech adopts the two-pass architecture to decompose the generation process into two steps, where a linguistic decoder first generates the target text, and an acoustic decoder then generates the target speech based on the hidden states of the linguistic decoder. Specifically, we use the decoder of DA-Transformer as the linguistic decoder, and use FastSpeech 2 as the acoustic decoder. DA-Transformer models translations with a directed acyclic graph (DAG). To consider all potential paths in the DAG during training, we calculate the expected hidden states for each target token via dynamic programming, and feed them into the acoustic decoder to predict the target mel-spectrogram. During inference, we select the most probable path and take hidden states on that path as input to the acoustic decoder. Experiments on the CVSS Fr-En benchmark demonstrate that DASpeech can achieve comparable or even better performance than the state-of-the-art S2ST model Translatotron 2, while preserving up to 18.53x speedup compared to the autoregressive baseline. Compared with the previous non-autoregressive S2ST model, DASpeech does not rely on knowledge distillation and iterative decoding, achieving significant improvements in both translation quality and decoding speed. Furthermore, DASpeech shows the ability to preserve the speaker's voice of the source speech during translation.

{{</citation>}}


### (69/185) Target-oriented Proactive Dialogue Systems with Personalization: Problem Formulation and Dataset Curation (Jian Wang et al., 2023)

{{<citation>}}

Jian Wang, Yi Cheng, Dongding Lin, Chak Tou Leong, Wenjie Li. (2023)  
**Target-oriented Proactive Dialogue Systems with Personalization: Problem Formulation and Dataset Curation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.07397v2)  

---


**ABSTRACT**  
Target-oriented dialogue systems, designed to proactively steer conversations toward predefined targets or accomplish specific system-side goals, are an exciting area in conversational AI. In this work, by formulating a <dialogue act, topic> pair as the conversation target, we explore a novel problem of personalized target-oriented dialogue by considering personalization during the target accomplishment process. However, there remains an emergent need for high-quality datasets, and building one from scratch requires tremendous human effort. To address this, we propose an automatic dataset curation framework using a role-playing approach. Based on this framework, we construct a large-scale personalized target-oriented dialogue dataset, TopDial, which comprises about 18K multi-turn dialogues. The experimental results show that this dataset is of high quality and could contribute to exploring personalized target-oriented dialogue.

{{</citation>}}


### (70/185) Investigating the Effect of Language Models in Sequence Discriminative Training for Neural Transducers (Zijian Yang et al., 2023)

{{<citation>}}

Zijian Yang, Wei Zhou, Ralf Schlüter, Hermann Ney. (2023)  
**Investigating the Effect of Language Models in Sequence Discriminative Training for Neural Transducers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07345v1)  

---


**ABSTRACT**  
In this work, we investigate the effect of language models (LMs) with different context lengths and label units (phoneme vs. word) used in sequence discriminative training for phoneme-based neural transducers. Both lattice-free and N-best-list approaches are examined. For lattice-free methods with phoneme-level LMs, we propose a method to approximate the context history to employ LMs with full-context dependency. This approximation can be extended to arbitrary context length and enables the usage of word-level LMs in lattice-free methods. Moreover, a systematic comparison is conducted across lattice-free and N-best-list-based methods. Experimental results on Librispeech show that using the word-level LM in training outperforms the phoneme-level LM. Besides, we find that the context size of the LM used for probability computation has a limited effect on performance. Moreover, our results reveal the pivotal importance of the hypothesis space quality in sequence discriminative training.

{{</citation>}}


### (71/185) How Do Large Language Models Capture the Ever-changing World Knowledge? A Review of Recent Advances (Zihan Zhang et al., 2023)

{{<citation>}}

Zihan Zhang, Meng Fang, Ling Chen, Mohammad-Reza Namazi-Rad, Jun Wang. (2023)  
**How Do Large Language Models Capture the Ever-changing World Knowledge? A Review of Recent Advances**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07343v1)  

---


**ABSTRACT**  
Although large language models (LLMs) are impressive in solving various tasks, they can quickly be outdated after deployment. Maintaining their up-to-date status is a pressing concern in the current era. This paper provides a comprehensive review of recent advances in aligning LLMs with the ever-changing world knowledge without re-training from scratch. We categorize research works systemically and provide in-depth comparisons and discussion. We also discuss existing challenges and highlight future directions to facilitate research in this field. We release the paper list at https://github.com/hyintell/awesome-refreshing-llms

{{</citation>}}


### (72/185) An Empirical Study of Instruction-tuning Large Language Models in Chinese (Qingyi Si et al., 2023)

{{<citation>}}

Qingyi Si, Tong Wang, Zheng Lin, Xu Zhang, Yanan Cao, Weiping Wang. (2023)  
**An Empirical Study of Instruction-tuning Large Language Models in Chinese**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GLM, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07328v1)  

---


**ABSTRACT**  
The success of ChatGPT validates the potential of large language models (LLMs) in artificial general intelligence (AGI). Subsequently, the release of LLMs has sparked the open-source community's interest in instruction-tuning, which is deemed to accelerate ChatGPT's replication process. However, research on instruction-tuning LLMs in Chinese, the world's most spoken language, is still in its early stages. Therefore, this paper makes an in-depth empirical study of instruction-tuning LLMs in Chinese, which can serve as a cookbook that provides valuable findings for effectively customizing LLMs that can better respond to Chinese instructions. Specifically, we systematically explore the impact of LLM bases, parameter-efficient methods, instruction data types, which are the three most important elements for instruction-tuning. Besides, we also conduct experiment to study the impact of other factors, e.g., chain-of-thought data and human-value alignment. We hope that this empirical study can make a modest contribution to the open Chinese version of ChatGPT. This paper will release a powerful Chinese LLMs that is comparable to ChatGLM. The code and data are available at https://github.com/PhoebusSi/Alpaca-CoT.

{{</citation>}}


### (73/185) On the Impact of Cross-Domain Data on German Language Models (Amin Dada et al., 2023)

{{<citation>}}

Amin Dada, Aokun Chen, Cheng Peng, Kaleb E Smith, Ahmad Idrissi-Yaghir, Constantin Marc Seibold, Jianning Li, Lars Heiliger, Xi Yang, Christoph M. Friedrich, Daniel Truhn, Jan Egger, Jiang Bian, Jens Kleesiek, Yonghui Wu. (2023)  
**On the Impact of Cross-Domain Data on German Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07321v2)  

---


**ABSTRACT**  
Traditionally, large language models have been either trained on general web crawls or domain-specific data. However, recent successes of generative large language models, have shed light on the benefits of cross-domain datasets. To examine the significance of prioritizing data diversity over quality, we present a German dataset comprising texts from five domains, along with another dataset aimed at containing high-quality data. Through training a series of models ranging between 122M and 750M parameters on both datasets, we conduct a comprehensive benchmark on multiple downstream tasks. Our findings demonstrate that the models trained on the cross-domain dataset outperform those trained on quality data alone, leading to improvements up to $4.45\%$ over the previous state-of-the-art. The models are available at https://huggingface.co/ikim-uk-essen

{{</citation>}}


### (74/185) Parrot: Enhancing Multi-Turn Chat Models by Learning to Ask Questions (Yuchong Sun et al., 2023)

{{<citation>}}

Yuchong Sun, Che Liu, Jinwen Huang, Ruihua Song, Fuzheng Zhang, Di Zhang, Zhongyuan Wang, Kun Gai. (2023)  
**Parrot: Enhancing Multi-Turn Chat Models by Learning to Ask Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07301v1)  

---


**ABSTRACT**  
Impressive progress has been made on chat models based on Large Language Models (LLMs) recently; however, there is a noticeable lag in multi-turn conversations between open-source chat models (e.g., Alpaca and Vicuna) and the leading chat models (e.g., ChatGPT and GPT-4). Through a series of analyses, we attribute the lag to the lack of enough high-quality multi-turn instruction-tuning data. The available instruction-tuning data for the community are either single-turn conversations or multi-turn ones with certain issues, such as non-human-like instructions, less detailed responses, or rare topic shifts. In this paper, we address these challenges by introducing Parrot, a highly scalable solution designed to automatically generate high-quality instruction-tuning data, which are then used to enhance the effectiveness of chat models in multi-turn conversations. Specifically, we start by training the Parrot-Ask model, which is designed to emulate real users in generating instructions. We then utilize Parrot-Ask to engage in multi-turn conversations with ChatGPT across a diverse range of topics, resulting in a collection of 40K high-quality multi-turn dialogues (Parrot-40K). These data are subsequently employed to train a chat model that we have named Parrot-Chat. We demonstrate that the dialogues gathered from Parrot-Ask markedly outperform existing multi-turn instruction-following datasets in critical metrics, including topic diversity, number of turns, and resemblance to human conversation. With only 40K training examples, Parrot-Chat achieves strong performance against other 13B open-source models across a range of instruction-following benchmarks, and particularly excels in evaluations of multi-turn capabilities. We make all codes, datasets, and two versions of the Parrot-Ask model based on LLaMA2-13B and KuaiYii-13B available at https://github.com/kwai/KwaiYii/Parrot.

{{</citation>}}


### (75/185) Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators (Liang Chen et al., 2023)

{{<citation>}}

Liang Chen, Yang Deng, Yatao Bian, Zeyu Qin, Bingzhe Wu, Tat-Seng Chua, Kam-Fai Wong. (2023)  
**Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NER  
[Paper Link](http://arxiv.org/abs/2310.07289v1)  

---


**ABSTRACT**  
Large language models (LLMs) outperform information retrieval techniques for downstream knowledge-intensive tasks when being prompted to generate world knowledge. However, community concerns abound regarding the factuality and potential implications of using this uncensored knowledge. In light of this, we introduce CONNER, a COmpreheNsive kNowledge Evaluation fRamework, designed to systematically and automatically evaluate generated knowledge from six important perspectives -- Factuality, Relevance, Coherence, Informativeness, Helpfulness and Validity. We conduct an extensive empirical analysis of the generated knowledge from three different types of LLMs on two widely studied knowledge-intensive tasks, i.e., open-domain question answering and knowledge-grounded dialogue. Surprisingly, our study reveals that the factuality of generated knowledge, even if lower, does not significantly hinder downstream tasks. Instead, the relevance and coherence of the outputs are more important than small factual mistakes. Further, we show how to use CONNER to improve knowledge-intensive tasks by designing two strategies: Prompt Engineering and Knowledge Selection. Our evaluation code and LLM-generated knowledge with human annotations will be released to facilitate future research.

{{</citation>}}


### (76/185) BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations (Qizhi Pei et al., 2023)

{{<citation>}}

Qizhi Pei, Wei Zhang, Jinhua Zhu, Kehan Wu, Kaiyuan Gao, Lijun Wu, Yingce Xia, Rui Yan. (2023)  
**BioT5: Enriching Cross-modal Integration in Biology with Chemical Knowledge and Natural Language Associations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, q-bio-BM  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2310.07276v1)  

---


**ABSTRACT**  
Recent advancements in biological research leverage the integration of molecules, proteins, and natural language to enhance drug discovery. However, current models exhibit several limitations, such as the generation of invalid molecular SMILES, underutilization of contextual information, and equal treatment of structured and unstructured knowledge. To address these issues, we propose $\mathbf{BioT5}$, a comprehensive pre-training framework that enriches cross-modal integration in biology with chemical knowledge and natural language associations. $\mathbf{BioT5}$ utilizes SELFIES for $100%$ robust molecular representations and extracts knowledge from the surrounding context of bio-entities in unstructured biological literature. Furthermore, $\mathbf{BioT5}$ distinguishes between structured and unstructured knowledge, leading to more effective utilization of information. After fine-tuning, BioT5 shows superior performance across a wide range of tasks, demonstrating its strong capability of capturing underlying relations and properties of bio-entities. Our code is available at $\href{https://github.com/QizhiPei/BioT5}{Github}$.

{{</citation>}}


### (77/185) Ethical Reasoning over Moral Alignment: A Case and Framework for In-Context Ethical Policies in LLMs (Abhinav Rao et al., 2023)

{{<citation>}}

Abhinav Rao, Aditi Khandelwal, Kumar Tanmay, Utkarsh Agarwal, Monojit Choudhury. (2023)  
**Ethical Reasoning over Moral Alignment: A Case and Framework for In-Context Ethical Policies in LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.07251v1)  

---


**ABSTRACT**  
In this position paper, we argue that instead of morally aligning LLMs to specific set of ethical principles, we should infuse generic ethical reasoning capabilities into them so that they can handle value pluralism at a global scale. When provided with an ethical policy, an LLM should be capable of making decisions that are ethically consistent to the policy. We develop a framework that integrates moral dilemmas with moral principles pertaining to different foramlisms of normative ethics, and at different levels of abstractions. Initial experiments with GPT-x models shows that while GPT-4 is a nearly perfect ethical reasoner, the models still have bias towards the moral values of Western and English speaking societies.

{{</citation>}}


### (78/185) Exploring the Landscape of Large Language Models In Medical Question Answering: Observations and Open Questions (Karolina Korgul et al., 2023)

{{<citation>}}

Karolina Korgul, Andrew M. Bean, Felix Krones, Robert McCraith, Adam Mahdi. (2023)  
**Exploring the Landscape of Large Language Models In Medical Question Answering: Observations and Open Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.07225v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown promise in medical question answering by achieving passing scores in standardised exams and have been suggested as tools for supporting healthcare workers. Deploying LLMs into such a high-risk context requires a clear understanding of the limitations of these models. With the rapid development and release of new LLMs, it is especially valuable to identify patterns which exist across models and may, therefore, continue to appear in newer versions. In this paper, we evaluate a wide range of popular LLMs on their knowledge of medical questions in order to better understand their properties as a group. From this comparison, we provide preliminary observations and raise open questions for further research.

{{</citation>}}


### (79/185) Adaptive Gating in Mixture-of-Experts based Language Models (Jiamin Li et al., 2023)

{{<citation>}}

Jiamin Li, Qiang Su, Yitao Yang, Yimin Jiang, Cong Wang, Hong Xu. (2023)  
**Adaptive Gating in Mixture-of-Experts based Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.07188v1)  

---


**ABSTRACT**  
Large language models, such as OpenAI's ChatGPT, have demonstrated exceptional language understanding capabilities in various NLP tasks. Sparsely activated mixture-of-experts (MoE) has emerged as a promising solution for scaling models while maintaining a constant number of computational operations. Existing MoE model adopts a fixed gating network where each token is computed by the same number of experts. However, this approach contradicts our intuition that the tokens in each sequence vary in terms of their linguistic complexity and, consequently, require different computational costs. Little is discussed in prior research on the trade-off between computation per token and model performance. This paper introduces adaptive gating in MoE, a flexible training strategy that allows tokens to be processed by a variable number of experts based on expert probability distribution. The proposed framework preserves sparsity while improving training efficiency. Additionally, curriculum learning is leveraged to further reduce training time. Extensive experiments on diverse NLP tasks show that adaptive gating reduces at most 22.5% training time while maintaining inference quality. Moreover, we conduct a comprehensive analysis of the routing decisions and present our insights when adaptive gating is used.

{{</citation>}}


### (80/185) PHALM: Building a Knowledge Graph from Scratch by Prompting Humans and a Language Model (Tatsuya Ide et al., 2023)

{{<citation>}}

Tatsuya Ide, Eiki Murata, Daisuke Kawahara, Takato Yamazaki, Shengzhe Li, Kenta Shinzato, Toshinori Sato. (2023)  
**PHALM: Building a Knowledge Graph from Scratch by Prompting Humans and a Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07170v1)  

---


**ABSTRACT**  
Despite the remarkable progress in natural language understanding with pretrained Transformers, neural language models often do not handle commonsense knowledge well. Toward commonsense-aware models, there have been attempts to obtain knowledge, ranging from automatic acquisition to crowdsourcing. However, it is difficult to obtain a high-quality knowledge base at a low cost, especially from scratch. In this paper, we propose PHALM, a method of building a knowledge graph from scratch, by prompting both crowdworkers and a large language model (LLM). We used this method to build a Japanese event knowledge graph and trained Japanese commonsense generation models. Experimental results revealed the acceptability of the built graph and inferences generated by the trained models. We also report the difference in prompting humans and an LLM. Our code, data, and models are available at github.com/nlp-waseda/comet-atomic-ja.

{{</citation>}}


### (81/185) QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources (Zhikai Li et al., 2023)

{{<citation>}}

Zhikai Li, Xiaoxuan Liu, Banghua Zhu, Zhen Dong, Qingyi Gu, Kurt Keutzer. (2023)  
**QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07147v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pre-trained models on downstream datasets provides further significant performance gains, but this process has been challenging due to its extraordinary resource requirements. To this end, existing efforts focus on parameter-efficient fine-tuning, which, unfortunately, fail to capitalize on the powerful potential of full-parameter fine-tuning. In this work, we propose QFT, a novel Quantized Full-parameter Tuning framework for LLMs that enables memory-efficient fine-tuning without harming performance. Our framework incorporates two novel ideas: (i) we adopt the efficient Lion optimizer, which only keeps track of the momentum and has consistent update magnitudes for each parameter, an inherent advantage for robust quantization; and (ii) we quantize all model states and store them as integer values, and present a gradient flow and parameter update scheme for the quantized weights. As a result, QFT reduces the model state memory to 21% of the standard solution while achieving comparable performance, e.g., tuning a LLaMA-7B model requires only <30GB of memory, satisfied by a single A6000 GPU.

{{</citation>}}


### (82/185) Empowering Psychotherapy with Large Language Models: Cognitive Distortion Detection through Diagnosis of Thought Prompting (Zhiyu Chen et al., 2023)

{{<citation>}}

Zhiyu Chen, Yujie Lu, William Yang Wang. (2023)  
**Empowering Psychotherapy with Large Language Models: Cognitive Distortion Detection through Diagnosis of Thought Prompting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07146v1)  

---


**ABSTRACT**  
Mental illness remains one of the most critical public health issues of our time, due to the severe scarcity and accessibility limit of professionals. Psychotherapy requires high-level expertise to conduct deep, complex reasoning and analysis on the cognition modeling of the patients. In the era of Large Language Models, we believe it is the right time to develop AI assistance for computational psychotherapy. We study the task of cognitive distortion detection and propose the Diagnosis of Thought (DoT) prompting. DoT performs diagnosis on the patient's speech via three stages: subjectivity assessment to separate the facts and the thoughts; contrastive reasoning to elicit the reasoning processes supporting and contradicting the thoughts; and schema analysis to summarize the cognition schemas. The generated diagnosis rationales through the three stages are essential for assisting the professionals. Experiments demonstrate that DoT obtains significant improvements over ChatGPT for cognitive distortion detection, while generating high-quality rationales approved by human experts.

{{</citation>}}


### (83/185) The Temporal Structure of Language Processing in the Human Brain Corresponds to The Layered Hierarchy of Deep Language Models (Ariel Goldstein et al., 2023)

{{<citation>}}

Ariel Goldstein, Eric Ham, Mariano Schain, Samuel Nastase, Zaid Zada, Avigail Dabush, Bobbi Aubrey, Harshvardhan Gazula, Amir Feder, Werner K Doyle, Sasha Devore, Patricia Dugan, Daniel Friedman, Roi Reichart, Michael Brenner, Avinatan Hassidim, Orrin Devinsky, Adeen Flinker, Omer Levy, Uri Hasson. (2023)  
**The Temporal Structure of Language Processing in the Human Brain Corresponds to The Layered Hierarchy of Deep Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, q-bio-NC  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07106v1)  

---


**ABSTRACT**  
Deep Language Models (DLMs) provide a novel computational paradigm for understanding the mechanisms of natural language processing in the human brain. Unlike traditional psycholinguistic models, DLMs use layered sequences of continuous numerical vectors to represent words and context, allowing a plethora of emerging applications such as human-like text generation. In this paper we show evidence that the layered hierarchy of DLMs may be used to model the temporal dynamics of language comprehension in the brain by demonstrating a strong correlation between DLM layer depth and the time at which layers are most predictive of the human brain. Our ability to temporally resolve individual layers benefits from our use of electrocorticography (ECoG) data, which has a much higher temporal resolution than noninvasive methods like fMRI. Using ECoG, we record neural activity from participants listening to a 30-minute narrative while also feeding the same narrative to a high-performing DLM (GPT2-XL). We then extract contextual embeddings from the different layers of the DLM and use linear encoding models to predict neural activity. We first focus on the Inferior Frontal Gyrus (IFG, or Broca's area) and then extend our model to track the increasing temporal receptive window along the linguistic processing hierarchy from auditory to syntactic and semantic areas. Our results reveal a connection between human language processing and DLMs, with the DLM's layer-by-layer accumulation of contextual information mirroring the timing of neural activity in high-order language areas.

{{</citation>}}


### (84/185) Sparse Universal Transformer (Shawn Tan et al., 2023)

{{<citation>}}

Shawn Tan, Yikang Shen, Zhenfang Chen, Aaron Courville, Chuang Gan. (2023)  
**Sparse Universal Transformer**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07096v1)  

---


**ABSTRACT**  
The Universal Transformer (UT) is a variant of the Transformer that shares parameters across its layers. Empirical evidence shows that UTs have better compositional generalization than Vanilla Transformers (VTs) in formal language tasks. The parameter-sharing also affords it better parameter efficiency than VTs. Despite its many advantages, scaling UT parameters is much more compute and memory intensive than scaling up a VT. This paper proposes the Sparse Universal Transformer (SUT), which leverages Sparse Mixture of Experts (SMoE) and a new stick-breaking-based dynamic halting mechanism to reduce UT's computation complexity while retaining its parameter efficiency and generalization ability. Experiments show that SUT achieves the same performance as strong baseline models while only using half computation and parameters on WMT'14 and strong generalization results on formal language tasks (Logical inference and CFQ). The new halting mechanism also enables around 50\% reduction in computation during inference with very little performance decrease on formal language tasks.

{{</citation>}}


### (85/185) Argumentative Stance Prediction: An Exploratory Study on Multimodality and Few-Shot Learning (Arushi Sharma et al., 2023)

{{<citation>}}

Arushi Sharma, Abhibha Gupta, Maneesh Bilalpur. (2023)  
**Argumentative Stance Prediction: An Exploratory Study on Multimodality and Few-Shot Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.07093v1)  

---


**ABSTRACT**  
To advance argumentative stance prediction as a multimodal problem, the First Shared Task in Multimodal Argument Mining hosted stance prediction in crucial social topics of gun control and abortion. Our exploratory study attempts to evaluate the necessity of images for stance prediction in tweets and compare out-of-the-box text-based large-language models (LLM) in few-shot settings against fine-tuned unimodal and multimodal models. Our work suggests an ensemble of fine-tuned text-based language models (0.817 F1-score) outperforms both the multimodal (0.677 F1-score) and text-based few-shot prediction using a recent state-of-the-art LLM (0.550 F1-score). In addition to the differences in performance, our findings suggest that the multimodal models tend to perform better when image content is summarized as natural language over their native pixel structure and, using in-context examples improves few-shot performance of LLMs.

{{</citation>}}


### (86/185) Jaeger: A Concatenation-Based Multi-Transformer VQA Model (Jieting Long et al., 2023)

{{<citation>}}

Jieting Long, Zewei Shi, Penghao Jiang, Yidong Gan. (2023)  
**Jaeger: A Concatenation-Based Multi-Transformer VQA Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT, QA, Question Answering, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07091v1)  

---


**ABSTRACT**  
Document-based Visual Question Answering poses a challenging task between linguistic sense disambiguation and fine-grained multimodal retrieval. Although there has been encouraging progress in document-based question answering due to the utilization of large language and open-world prior models\cite{1}, several challenges persist, including prolonged response times, extended inference durations, and imprecision in matching. In order to overcome these challenges, we propose Jaegar, a concatenation-based multi-transformer VQA model. To derive question features, we leverage the exceptional capabilities of RoBERTa large\cite{2} and GPT2-xl\cite{3} as feature extractors. Subsequently, we subject the outputs from both models to a concatenation process. This operation allows the model to consider information from diverse sources concurrently, strengthening its representational capability. By leveraging pre-trained models for feature extraction, our approach has the potential to amplify the performance of these models through concatenation. After concatenation, we apply dimensionality reduction to the output features, reducing the model's computational effectiveness and inference time. Empirical results demonstrate that our proposed model achieves competitive performance on Task C of the PDF-VQA Dataset. If the user adds any new data, they should make sure to style it as per the instructions provided in previous sections.

{{</citation>}}


### (87/185) Diversity of Thought Improves Reasoning Abilities of Large Language Models (Ranjita Naik et al., 2023)

{{<citation>}}

Ranjita Naik, Varun Chandrasekaran, Mert Yuksekgonul, Hamid Palangi, Besmira Nushi. (2023)  
**Diversity of Thought Improves Reasoning Abilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.07088v1)  

---


**ABSTRACT**  
Large language models (LLMs) are documented to struggle in settings that require complex reasoning. Nevertheless, instructing the model to break down the problem into smaller reasoning steps (Wei et al., 2022), or ensembling various generations through modifying decoding steps (Wang et al., 2023) boosts performance. Current methods assume that the input prompt is fixed and expect the decoding strategies to introduce the diversity needed for ensembling. In this work, we relax this assumption and discuss how one can create and leverage variations of the input prompt as a means to diversity of thought to improve model performance. We propose a method that automatically improves prompt diversity by soliciting feedback from the LLM to ideate approaches that fit for the problem. We then ensemble the diverse prompts in our method DIV-SE (DIVerse reasoning path Self-Ensemble) across multiple inference calls. We also propose a cost-effective alternative where diverse prompts are used within a single inference call; we call this IDIV-SE (In-call DIVerse reasoning path Self-Ensemble). Under a fixed generation budget, DIV-SE and IDIV-SE outperform the previously discussed baselines using both GPT-3.5 and GPT-4 on several reasoning benchmarks, without modifying the decoding process. Additionally, DIV-SE advances state-of-the-art performance on recent planning benchmarks (Valmeekam et al., 2023), exceeding the highest previously reported accuracy by at least 29.6 percentage points on the most challenging 4/5 Blocksworld task. Our results shed light on how to enforce prompt diversity toward LLM reasoning and thereby improve the pareto frontier of the accuracy-cost trade-off.

{{</citation>}}


## cs.SE (5)



### (88/185) A Large-Scale Exploratory Study of Android Sports Apps in the Google Play Store (Bhagya Chembakottu et al., 2023)

{{<citation>}}

Bhagya Chembakottu, Heng Li, Foutse Khomh. (2023)  
**A Large-Scale Exploratory Study of Android Sports Apps in the Google Play Store**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.07921v1)  

---


**ABSTRACT**  
Prior studies on mobile app analysis often analyze apps across different categories or focus on a small set of apps within a category. These studies either provide general insights for an entire app store which consists of millions of apps, or provide specific insights for a small set of apps. However, a single app category can often contain tens of thousands to hundreds of thousands of apps. For example, according to AppBrain, there are 46,625 apps in the "Sports" category of Google Play apps. Analyzing such a targeted category of apps can provide more specific insights than analyzing apps across categories while still benefiting many app developers interested in the category. This work aims to study a large number of apps from a single category (i.e., the sports category). We performed an empirical study on over two thousand sports apps in the Google Play Store. We study the characteristics of these apps (e.g., their targeted sports types and main functionalities) through manual analysis, the topics in the user review through topic modeling, as well as the aspects that contribute to the negative opinions of users through analysis of user ratings and sentiment. It is concluded that analyzing a targeted category of apps (e.g., sports apps) can provide more specific insights than analyzing apps across different categories while still being relevant for a large number (e.g., tens of thousands) of apps. Besides, as a rapid-growing and competitive market, sports apps provide rich opportunities for future research, for example, to study the integration of data science or machine learning techniques in software applications or to study the factors that influence the competitiveness of the apps.

{{</citation>}}


### (89/185) RLaGA: A Reinforcement Learning Augmented Genetic Algorithm For Searching Real and Diverse Marker-Based Landing Violations (Linfeng Liang et al., 2023)

{{<citation>}}

Linfeng Liang, Yao Deng, Kye Morton, Valtteri Kallinen, Alice James, Avishkar Seth, Endrowednes Kuantama, Subhas Mukhopadhyay, Richard Han, Xi Zheng. (2023)  
**RLaGA: A Reinforcement Learning Augmented Genetic Algorithm For Searching Real and Diverse Marker-Based Landing Violations**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07378v2)  

---


**ABSTRACT**  
Automated landing for Unmanned Aerial Vehicles (UAVs), like multirotor drones, requires intricate software encompassing control algorithms, obstacle avoidance, and machine vision, especially when landing markers assist. Failed landings can lead to significant costs from damaged drones or payloads and the time spent seeking alternative landing solutions. Therefore, it's important to fully test auto-landing systems through simulations before deploying them in the real-world to ensure safety. This paper proposes RLaGA, a reinforcement learning (RL) augmented search-based testing framework, which constructs diverse and real marker-based landing cases that involve safety violations. Specifically, RLaGA introduces a genetic algorithm (GA) to conservatively search for diverse static environment configurations offline and RL to aggressively manipulate dynamic objects' trajectories online to find potential vulnerabilities in the target deployment environment. Quantitative results reveal that our method generates up to 22.19% more violation cases and nearly doubles the diversity of generated violation cases compared to baseline methods. Qualitatively, our method can discover those corner cases which would be missed by state-of-the-art algorithms. We demonstrate that select types of these corner cases can be confirmed via real-world testing with drones in the field.

{{</citation>}}


### (90/185) Revisiting Android App Categorization (Marco Alecci et al., 2023)

{{<citation>}}

Marco Alecci, Jordan Samhi, Tegawendé F. Bissyandé, Jacques Klein. (2023)  
**Revisiting Android App Categorization**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.07290v1)  

---


**ABSTRACT**  
Numerous tools rely on automatic categorization of Android apps as part of their methodology. However, incorrect categorization can lead to inaccurate outcomes, such as a malware detector wrongly flagging a benign app as malicious. One such example is the SlideIT Free Keyboard app, which has over 500000 downloads on Google Play. Despite being a "Keyboard" app, it is often wrongly categorized alongside "Language" apps due to the app's description focusing heavily on language support, resulting in incorrect analysis outcomes, including mislabeling it as a potential malware when it is actually a benign app. Hence, there is a need to improve the categorization of Android apps to benefit all the tools relying on it. In this paper, we present a comprehensive evaluation of existing Android app categorization approaches using our new ground-truth dataset. Our evaluation demonstrates the notable superiority of approaches that utilize app descriptions over those solely relying on data extracted from the APK file, while also leaving space for potential improvement in the former category. Thus, we propose two innovative approaches that effectively outperform the performance of existing methods in both description-based and APK-based methodologies. Finally, by employing our novel description-based approach, we have successfully demonstrated that adopting a higher-performing categorization method can significantly benefit tools reliant on app categorization, leading to an improvement in their overall performance. This highlights the significance of developing advanced and efficient app categorization methodologies for improved results in software engineering tasks.

{{</citation>}}


### (91/185) CrashTranslator: Automatically Reproducing Mobile Application Crashes Directly from Stack Trace (Yuchao Huang et al., 2023)

{{<citation>}}

Yuchao Huang, Junjie Wang, Zhe Liu, Yawen Wang, Song Wang, Chunyang Chen, Yuanzhe Hu, Qing Wang. (2023)  
**CrashTranslator: Automatically Reproducing Mobile Application Crashes Directly from Stack Trace**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07128v1)  

---


**ABSTRACT**  
Crash reports are vital for software maintenance since they allow the developers to be informed of the problems encountered in the mobile application. Before fixing, developers need to reproduce the crash, which is an extremely time-consuming and tedious task. Existing studies conducted the automatic crash reproduction with the natural language described reproducing steps. Yet we find a non-neglectable portion of crash reports only contain the stack trace when the crash occurs. Such stack-trace-only crashes merely reveal the last GUI page when the crash occurs, and lack step-by-step guidance. Developers tend to spend more effort in understanding the problem and reproducing the crash, and existing techniques cannot work on this, thus calling for a greater need for automatic support. This paper proposes an approach named CrashTranslator to automatically reproduce mobile application crashes directly from the stack trace. It accomplishes this by leveraging a pre-trained Large Language Model to predict the exploration steps for triggering the crash, and designing a reinforcement learning based technique to mitigate the inaccurate prediction and guide the search holistically. We evaluate CrashTranslator on 75 crash reports involving 58 popular Android apps, and it successfully reproduces 61.3% of the crashes, outperforming the state-of-the-art baselines by 109% to 206%. Besides, the average reproducing time is 68.7 seconds, outperforming the baselines by 302% to 1611%. We also evaluate the usefulness of CrashTranslator with promising results.

{{</citation>}}


### (92/185) SparseCoder: Advancing Source Code Analysis with Sparse Attention and Learned Token Pruning (Xueqi Yang et al., 2023)

{{<citation>}}

Xueqi Yang, Mariusz Jakubowski, Kelly Kang, Haojie Yu, Tim Menzies. (2023)  
**SparseCoder: Advancing Source Code Analysis with Sparse Attention and Learned Token Pruning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Attention, BERT, Pruning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07109v1)  

---


**ABSTRACT**  
As software projects rapidly evolve, software artifacts become more complex and defects behind get harder to identify. The emerging Transformer-based approaches, though achieving remarkable performance, struggle with long code sequences due to their self-attention mechanism, which scales quadratically with the sequence length. This paper introduces SparseCoder, an innovative approach incorporating sparse attention and learned token pruning (LTP) method (adapted from natural language processing) to address this limitation. Extensive experiments carried out on a large-scale dataset for vulnerability detection demonstrate the effectiveness and efficiency of SparseCoder, scaling from quadratically to linearly on long code sequence analysis in comparison to CodeBERT and RoBERTa. We further achieve 50% FLOPs reduction with a negligible performance drop of less than 1% comparing to Transformer leveraging sparse attention. Moverover, SparseCoder goes beyond making "black-box" decisions by elucidating the rationale behind those decisions. Code segments that contribute to the final decision can be highlighted with importance scores, offering an interpretable, transparent analysis tool for the software engineering landscape.

{{</citation>}}


## cs.NI (4)



### (93/185) Tag Your Fish in the Broken Net: A Responsible Web Framework for Protecting Online Privacy and Copyright (Dawen Zhang et al., 2023)

{{<citation>}}

Dawen Zhang, Boming Xia, Yue Liu, Xiwei Xu, Thong Hoang, Zhenchang Xing, Mark Staples, Qinghua Lu, Liming Zhu. (2023)  
**Tag Your Fish in the Broken Net: A Responsible Web Framework for Protecting Online Privacy and Copyright**  

---
Primary Category: cs.NI  
Categories: cs-CY, cs-NI, cs-SI, cs.NI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.07915v1)  

---


**ABSTRACT**  
The World Wide Web, a ubiquitous source of information, serves as a primary resource for countless individuals, amassing a vast amount of data from global internet users. However, this online data, when scraped, indexed, and utilized for activities like web crawling, search engine indexing, and, notably, AI model training, often diverges from the original intent of its contributors. The ascent of Generative AI has accentuated concerns surrounding data privacy and copyright infringement. Regrettably, the web's current framework falls short in facilitating pivotal actions like consent withdrawal or data copyright claims. While some companies offer voluntary measures, such as crawler access restrictions, these often remain inaccessible to individual users. To empower online users to exercise their rights and enable companies to adhere to regulations, this paper introduces a user-controlled consent tagging framework for online data. It leverages the extensibility of HTTP and HTML in conjunction with the decentralized nature of distributed ledger technology. With this framework, users have the ability to tag their online data at the time of transmission, and subsequently, they can track and request the withdrawal of consent for their data from the data holders. A proof-of-concept system is implemented, demonstrating the feasibility of the framework. This work holds significant potential for contributing to the reinforcement of user consent, privacy, and copyright on the modern internet and lays the groundwork for future insights into creating a more responsible and user-centric web ecosystem.

{{</citation>}}


### (94/185) DeePref: Deep Reinforcement Learning For Video Prefetching In Content Delivery Networks (Nawras Alkassab et al., 2023)

{{<citation>}}

Nawras Alkassab, Chin-Tser Huang, Tania Lorido Botran. (2023)  
**DeePref: Deep Reinforcement Learning For Video Prefetching In Content Delivery Networks**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07881v1)  

---


**ABSTRACT**  
Content Delivery Networks carry the majority of Internet traffic, and the increasing demand for video content as a major IP traffic across the Internet highlights the importance of caching and prefetching optimization algorithms. Prefetching aims to make data available in the cache before the requester places its request to reduce access time and improve the Quality of Experience on the user side. Prefetching is well investigated in operating systems, compiler instructions, in-memory cache, local storage systems, high-speed networks, and cloud systems. Traditional prefetching techniques are well adapted to a particular access pattern, but fail to adapt to sudden variations or randomization in workloads. This paper explores the use of reinforcement learning to tackle the changes in user access patterns and automatically adapt over time. To this end, we propose, DeePref, a Deep Reinforcement Learning agent for online video content prefetching in Content Delivery Networks. DeePref is a prefetcher implemented on edge networks and is agnostic to hardware design, operating systems, and applications. Our results show that DeePref DRQN, using a real-world dataset, achieves a 17% increase in prefetching accuracy and a 28% increase in prefetching coverage on average compared to baseline approaches that use video content popularity as a building block to statically or dynamically make prefetching decisions. We also study the possibility of transfer learning of statistical models from one edge network into another, where unseen user requests from unknown distribution are observed. In terms of transfer learning, the increase in prefetching accuracy and prefetching coverage are [$30%$, $10%$], respectively. Our source code will be available on Github.

{{</citation>}}


### (95/185) AI/ML-based Load Prediction in IEEE 802.11 Enterprise Networks (Francesc Wilhelmi et al., 2023)

{{<citation>}}

Francesc Wilhelmi, Dariush Salami, Gianluca Fontanesi, Lorenzo Galati-Giordano, Mika Kasslin. (2023)  
**AI/ML-based Load Prediction in IEEE 802.11 Enterprise Networks**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07467v1)  

---


**ABSTRACT**  
Enterprise Wi-Fi networks can greatly benefit from Artificial Intelligence and Machine Learning (AI/ML) thanks to their well-developed management and operation capabilities. At the same time, AI/ML-based traffic/load prediction is one of the most appealing data-driven solutions to improve the Wi-Fi experience, either through the enablement of autonomous operation or by boosting troubleshooting with forecasted network utilization. In this paper, we study the suitability and feasibility of adopting AI/ML-based load prediction in practical enterprise Wi-Fi networks. While leveraging AI/ML solutions can potentially contribute to optimizing Wi-Fi networks in terms of energy efficiency, performance, and reliability, their effective adoption is constrained to aspects like data availability and quality, computational capabilities, and energy consumption. Our results show that hardware-constrained AI/ML models can potentially predict network load with less than 20% average error and 3% 85th-percentile error, which constitutes a suitable input for proactively driving Wi-Fi network optimization.

{{</citation>}}


### (96/185) CacheGen: Fast Context Loading for Language Model Applications (Yuhan Liu et al., 2023)

{{<citation>}}

Yuhan Liu, Hanchen Li, Kuntai Du, Jiayi Yao, Yihua Cheng, Yuyang Huang, Shan Lu, Michael Maire, Henry Hoffmann, Ari Holtzman, Ganesh Ananthanarayanan, Junchen Jiang. (2023)  
**CacheGen: Fast Context Loading for Language Model Applications**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07240v1)  

---


**ABSTRACT**  
As large language models (LLMs) take on more complex tasks, their inputs incorporate longer contexts to respond to questions that require domain knowledge or user-specific conversational histories. Yet, using long contexts poses a challenge for responsive LLM systems, as nothing can be generated until all the contexts are fetched to and processed by the LLM. Existing systems optimize only the computation delay in context processing (e.g., by caching intermediate key-value features of the text context) but often cause longer network delays in context fetching (e.g., key-value features consume orders of magnitude larger bandwidth than the text context).   This paper presents CacheGen to minimize the delays in fetching and processing contexts for LLMs. CacheGen reduces the bandwidth needed for transmitting long contexts' key-value (KV) features through a novel encoder that compresses KV features into more compact bitstream representations. The encoder combines adaptive quantization with a tailored arithmetic coder, taking advantage of the KV features' distributional properties, such as locality across tokens. Furthermore, CacheGen minimizes the total delay in fetching and processing a context by using a controller that determines when to load the context as compressed KV features or raw text and picks the appropriate compression level if loaded as KV features. We test CacheGen on three models of various sizes and three datasets of different context lengths. Compared to recent methods that handle long contexts, CacheGen reduces bandwidth usage by 3.7-4.3x and the total delay in fetching and processing contexts by 2.7-3x while maintaining similar LLM performance on various tasks as loading the text contexts.

{{</citation>}}


## cs.AI (20)



### (97/185) RoboCLIP: One Demonstration is Enough to Learn Robot Policies (Sumedh A Sontakke et al., 2023)

{{<citation>}}

Sumedh A Sontakke, Jesse Zhang, Sébastien M. R. Arnold, Karl Pertsch, Erdem Bıyık, Dorsa Sadigh, Chelsea Finn, Laurent Itti. (2023)  
**RoboCLIP: One Demonstration is Enough to Learn Robot Policies**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07899v1)  

---


**ABSTRACT**  
Reward specification is a notoriously difficult problem in reinforcement learning, requiring extensive expert supervision to design robust reward functions. Imitation learning (IL) methods attempt to circumvent these problems by utilizing expert demonstrations but typically require a large number of in-domain expert demonstrations. Inspired by advances in the field of Video-and-Language Models (VLMs), we present RoboCLIP, an online imitation learning method that uses a single demonstration (overcoming the large data requirement) in the form of a video demonstration or a textual description of the task to generate rewards without manual reward function design. Additionally, RoboCLIP can also utilize out-of-domain demonstrations, like videos of humans solving the task for reward generation, circumventing the need to have the same demonstration and deployment domains. RoboCLIP utilizes pretrained VLMs without any finetuning for reward generation. Reinforcement learning agents trained with RoboCLIP rewards demonstrate 2-3 times higher zero-shot performance than competing imitation learning methods on downstream robot manipulation tasks, doing so using only one video/text demonstration.

{{</citation>}}


### (98/185) Hierarchical Pretraining on Multimodal Electronic Health Records (Xiaochen Wang et al., 2023)

{{<citation>}}

Xiaochen Wang, Junyu Luo, Jiaqi Wang, Ziyi Yin, Suhan Cui, Yuan Zhong, Yaqing Wang, Fenglong Ma. (2023)  
**Hierarchical Pretraining on Multimodal Electronic Health Records**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.07871v1)  

---


**ABSTRACT**  
Pretraining has proven to be a powerful technique in natural language processing (NLP), exhibiting remarkable success in various NLP downstream tasks. However, in the medical domain, existing pretrained models on electronic health records (EHR) fail to capture the hierarchical nature of EHR data, limiting their generalization capability across diverse downstream tasks using a single pretrained model. To tackle this challenge, this paper introduces a novel, general, and unified pretraining framework called MEDHMP, specifically designed for hierarchically multimodal EHR data. The effectiveness of the proposed MEDHMP is demonstrated through experimental results on eight downstream tasks spanning three levels. Comparisons against eighteen baselines further highlight the efficacy of our approach.

{{</citation>}}


### (99/185) An Information Bottleneck Characterization of the Understanding-Workload Tradeoff (Lindsay Sanneman et al., 2023)

{{<citation>}}

Lindsay Sanneman, Mycal Tucker, Julie Shah. (2023)  
**An Information Bottleneck Characterization of the Understanding-Workload Tradeoff**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07802v1)  

---


**ABSTRACT**  
Recent advances in artificial intelligence (AI) have underscored the need for explainable AI (XAI) to support human understanding of AI systems. Consideration of human factors that impact explanation efficacy, such as mental workload and human understanding, is central to effective XAI design. Existing work in XAI has demonstrated a tradeoff between understanding and workload induced by different types of explanations. Explaining complex concepts through abstractions (hand-crafted groupings of related problem features) has been shown to effectively address and balance this workload-understanding tradeoff. In this work, we characterize the workload-understanding balance via the Information Bottleneck method: an information-theoretic approach which automatically generates abstractions that maximize informativeness and minimize complexity. In particular, we establish empirical connections between workload and complexity and between understanding and informativeness through human-subject experiments. This empirical link between human factors and information-theoretic concepts provides an important mathematical characterization of the workload-understanding tradeoff which enables user-tailored XAI design.

{{</citation>}}


### (100/185) Explainable Attention for Few-shot Learning and Beyond (Bahareh Nikpour et al., 2023)

{{<citation>}}

Bahareh Nikpour, Narges Armanfard. (2023)  
**Explainable Attention for Few-shot Learning and Beyond**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2310.07800v1)  

---


**ABSTRACT**  
Attention mechanisms have exhibited promising potential in enhancing learning models by identifying salient portions of input data. This is particularly valuable in scenarios where limited training samples are accessible due to challenges in data collection and labeling. Drawing inspiration from human recognition processes, we posit that an AI baseline's performance could be more accurate and dependable if it is exposed to essential segments of raw data rather than the entire input dataset, akin to human perception. However, the task of selecting these informative data segments, referred to as hard attention finding, presents a formidable challenge. In situations with few training samples, existing studies struggle to locate such informative regions due to the large number of training parameters that cannot be effectively learned from the available limited samples. In this study, we introduce a novel and practical framework for achieving explainable hard attention finding, specifically tailored for few-shot learning scenarios, called FewXAT. Our approach employs deep reinforcement learning to implement the concept of hard attention, directly impacting raw input data and thus rendering the process interpretable for human understanding. Through extensive experimentation across various benchmark datasets, we demonstrate the efficacy of our proposed method.

{{</citation>}}


### (101/185) SurroCBM: Concept Bottleneck Surrogate Models for Generative Post-hoc Explanation (Bo Pan et al., 2023)

{{<citation>}}

Bo Pan, Zhenke Liu, Yifei Zhang, Liang Zhao. (2023)  
**SurroCBM: Concept Bottleneck Surrogate Models for Generative Post-hoc Explanation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07698v1)  

---


**ABSTRACT**  
Explainable AI seeks to bring light to the decision-making processes of black-box models. Traditional saliency-based methods, while highlighting influential data segments, often lack semantic understanding. Recent advancements, such as Concept Activation Vectors (CAVs) and Concept Bottleneck Models (CBMs), offer concept-based explanations but necessitate human-defined concepts. However, human-annotated concepts are expensive to attain. This paper introduces the Concept Bottleneck Surrogate Models (SurroCBM), a novel framework that aims to explain the black-box models with automatically discovered concepts. SurroCBM identifies shared and unique concepts across various black-box models and employs an explainable surrogate model for post-hoc explanations. An effective training strategy using self-generated data is proposed to enhance explanation quality continuously. Through extensive experiments, we demonstrate the efficacy of SurroCBM in concept discovery and explanation, underscoring its potential in advancing the field of explainable AI.

{{</citation>}}


### (102/185) Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design (Lev Telyatnikov et al., 2023)

{{<citation>}}

Lev Telyatnikov, Maria Sofia Bucarelli, Guillermo Bernardez, Olga Zaghen, Simone Scardapane, Pietro Lio. (2023)  
**Hypergraph Neural Networks through the Lens of Message Passing: A Common Perspective to Homophily and Architecture Design**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SI, cs.AI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.07684v1)  

---


**ABSTRACT**  
Most of the current hypergraph learning methodologies and benchmarking datasets in the hypergraph realm are obtained by lifting procedures from their graph analogs, simultaneously leading to overshadowing hypergraph network foundations. This paper attempts to confront some pending questions in that regard: Can the concept of homophily play a crucial role in Hypergraph Neural Networks (HGNNs), similar to its significance in graph-based research? Is there room for improving current hypergraph architectures and methodologies? (e.g. by carefully addressing the specific characteristics of higher-order networks) Do existing datasets provide a meaningful benchmark for HGNNs? Diving into the details, this paper proposes a novel conceptualization of homophily in higher-order networks based on a message passing scheme; this approach harmonizes the analytical frameworks of datasets and architectures, offering a unified perspective for exploring and interpreting complex, higher-order network structures and dynamics. Further, we propose MultiSet, a novel message passing framework that redefines HGNNs by allowing hyperedge-dependent node representations, as well as introduce a novel architecture MultiSetMixer that leverages a new hyperedge sampling strategy. Finally, we provide an extensive set of experiments that contextualize our proposals and lead to valuable insights in hypergraph representation learning.

{{</citation>}}


### (103/185) Mini-DALLE3: Interactive Text to Image by Prompting Large Language Models (Zeqiang Lai et al., 2023)

{{<citation>}}

Zeqiang Lai, Xizhou Zhu, Jifeng Dai, Yu Qiao, Wenhai Wang. (2023)  
**Mini-DALLE3: Interactive Text to Image by Prompting Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07653v2)  

---


**ABSTRACT**  
The revolution of artificial intelligence content generation has been rapidly accelerated with the booming text-to-image (T2I) diffusion models. Within just two years of development, it was unprecedentedly of high-quality, diversity, and creativity that the state-of-the-art models could generate. However, a prevalent limitation persists in the effective communication with these popular T2I models, such as Stable Diffusion, using natural language descriptions. This typically makes an engaging image hard to obtain without expertise in prompt engineering with complex word compositions, magic tags, and annotations. Inspired by the recently released DALLE3 - a T2I model directly built-in ChatGPT that talks human language, we revisit the existing T2I systems endeavoring to align human intent and introduce a new task - interactive text to image (iT2I), where people can interact with LLM for interleaved high-quality image generation/edit/refinement and question answering with stronger images and text correspondences using natural language. In addressing the iT2I problem, we present a simple approach that augments LLMs for iT2I with prompting techniques and off-the-shelf T2I models. We evaluate our approach for iT2I in a variety of common-used scenarios under different LLMs, e.g., ChatGPT, LLAMA, Baichuan, and InternLM. We demonstrate that our approach could be a convenient and low-cost way to introduce the iT2I ability for any existing LLMs and any text-to-image models without any training while bringing little degradation on LLMs' inherent capabilities in, e.g., question answering and code generation. We hope this work could draw broader attention and provide inspiration for boosting user experience in human-machine interactions alongside the image quality of the next-generation T2I systems.

{{</citation>}}


### (104/185) Rethinking the BERT-like Pretraining for DNA Sequences (Chaoqi Liang et al., 2023)

{{<citation>}}

Chaoqi Liang, Weiqiang Bai, Lifeng Qiao, Yuchen Ren, Jianle Sun, Peng Ye, Hongliang Yan, Xinzhu Ma, Wangmeng Zuo, Wanli Ouyang. (2023)  
**Rethinking the BERT-like Pretraining for DNA Sequences**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2310.07644v2)  

---


**ABSTRACT**  
With the success of large-scale pretraining in NLP, there is an increasing trend of applying it to the domain of life sciences. In particular, pretraining methods based on DNA sequences have garnered growing attention due to their potential to capture generic information about genes. However, existing pretraining methods for DNA sequences largely rely on direct adoptions of BERT pretraining from NLP, lacking a comprehensive understanding and a specifically tailored approach. To address this research gap, we first conducted a series of exploratory experiments and gained several insightful observations: 1) In the fine-tuning phase of downstream tasks, when using K-mer overlapping tokenization instead of K-mer non-overlapping tokenization, both overlapping and non-overlapping pretraining weights show consistent performance improvement.2) During the pre-training process, using K-mer overlapping tokenization quickly produces clear K-mer embeddings and reduces the loss to a very low level, while using K-mer non-overlapping tokenization results in less distinct embeddings and continuously decreases the loss. 3) Using overlapping tokenization causes the self-attention in the intermediate layers of pre-trained models to tend to overly focus on certain tokens, reflecting that these layers are not adequately optimized. In summary, overlapping tokenization can benefit the fine-tuning of downstream tasks but leads to inadequate pretraining with fast convergence. To unleash the pretraining potential, we introduce a novel approach called RandomMask, which gradually increases the task difficulty of BERT-like pretraining by continuously expanding its mask boundary, forcing the model to learn more knowledge. RandomMask is simple but effective, achieving top-tier performance across 26 datasets of 28 datasets spanning 7 downstream tasks.

{{</citation>}}


### (105/185) OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models (Yuhe Liu et al., 2023)

{{<citation>}}

Yuhe Liu, Changhua Pei, Longlong Xu, Bohan Chen, Mingze Sun, Zhirui Zhang, Yongqian Sun, Shenglin Zhang, Kun Wang, Haiming Zhang, Jianhui Li, Gaogang Xie, Xidao Wen, Xiaohui Nie, Dan Pei. (2023)  
**OpsEval: A Comprehensive Task-Oriented AIOps Benchmark for Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NI, cs.AI  
Keywords: AI, GPT, Language Model, NLP, QA, Rouge  
[Paper Link](http://arxiv.org/abs/2310.07637v2)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited remarkable capabilities in NLP-related tasks such as translation, summarizing, and generation. The application of LLMs in specific areas, notably AIOps (Artificial Intelligence for IT Operations), holds great potential due to their advanced abilities in information summarizing, report analyzing, and ability of API calling. Nevertheless, the performance of current LLMs in AIOps tasks is yet to be determined. Furthermore, a comprehensive benchmark is required to steer the optimization of LLMs tailored for AIOps. Compared with existing benchmarks that focus on evaluating specific fields like network configuration, in this paper, we present \textbf{OpsEval}, a comprehensive task-oriented AIOps benchmark designed for LLMs. For the first time, OpsEval assesses LLMs' proficiency in three crucial scenarios (Wired Network Operation, 5G Communication Operation, and Database Operation) at various ability levels (knowledge recall, analytical thinking, and practical application). The benchmark includes 7,200 questions in both multiple-choice and question-answer (QA) formats, available in English and Chinese. With quantitative and qualitative results, we show how various LLM tricks can affect the performance of AIOps, including zero-shot, chain-of-thought, and few-shot in-context learning. We find that GPT4-score is more consistent with experts than widely used Bleu and Rouge, which can be used to replace automatic metrics for large-scale qualitative evaluations.

{{</citation>}}


### (106/185) Reinforcement Learning-based Knowledge Graph Reasoning for Explainable Fact-checking (Gustav Nikopensius et al., 2023)

{{<citation>}}

Gustav Nikopensius, Mohit Mayank, Orchid Chetia Phukan, Rajesh Sharma. (2023)  
**Reinforcement Learning-based Knowledge Graph Reasoning for Explainable Fact-checking**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: Knowledge Graph, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07613v1)  

---


**ABSTRACT**  
Fact-checking is a crucial task as it ensures the prevention of misinformation. However, manual fact-checking cannot keep up with the rate at which false information is generated and disseminated online. Automated fact-checking by machines is significantly quicker than by humans. But for better trust and transparency of these automated systems, explainability in the fact-checking process is necessary. Fact-checking often entails contrasting a factual assertion with a body of knowledge for such explanations. An effective way of representing knowledge is the Knowledge Graph (KG). There have been sufficient works proposed related to fact-checking with the usage of KG but not much focus is given to the application of reinforcement learning (RL) in such cases. To mitigate this gap, we propose an RL-based KG reasoning approach for explainable fact-checking. Extensive experiments on FB15K-277 and NELL-995 datasets reveal that reasoning over a KG is an effective way of producing human-readable explanations in the form of paths and classifications for fact claims. The RL reasoning agent computes a path that either proves or disproves a factual claim, but does not provide a verdict itself. A verdict is reached by a voting mechanism that utilizes paths produced by the agent. These paths can be presented to human readers so that they themselves can decide whether or not the provided evidence is convincing or not. This work will encourage works in this direction for incorporating RL for explainable fact-checking as it increases trustworthiness by providing a human-in-the-loop approach.

{{</citation>}}


### (107/185) Human-Centered Evaluation of XAI Methods (Karam Dawoud et al., 2023)

{{<citation>}}

Karam Dawoud, Wojciech Samek, Sebastian Lapuschkin, Sebastian Bosse. (2023)  
**Human-Centered Evaluation of XAI Methods**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07534v1)  

---


**ABSTRACT**  
In the ever-evolving field of Artificial Intelligence, a critical challenge has been to decipher the decision-making processes within the so-called "black boxes" in deep learning. Over recent years, a plethora of methods have emerged, dedicated to explaining decisions across diverse tasks. Particularly in tasks like image classification, these methods typically identify and emphasize the pivotal pixels that most influence a classifier's prediction. Interestingly, this approach mirrors human behavior: when asked to explain our rationale for classifying an image, we often point to the most salient features or aspects. Capitalizing on this parallel, our research embarked on a user-centric study. We sought to objectively measure the interpretability of three leading explanation methods: (1) Prototypical Part Network, (2) Occlusion, and (3) Layer-wise Relevance Propagation. Intriguingly, our results highlight that while the regions spotlighted by these methods can vary widely, they all offer humans a nearly equivalent depth of understanding. This enables users to discern and categorize images efficiently, reinforcing the value of these methods in enhancing AI transparency.

{{</citation>}}


### (108/185) Multimodal Graph Learning for Generative Tasks (Minji Yoon et al., 2023)

{{<citation>}}

Minji Yoon, Jing Yu Koh, Bryan Hooi, Ruslan Salakhutdinov. (2023)  
**Multimodal Graph Learning for Generative Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07478v2)  

---


**ABSTRACT**  
Multimodal learning combines multiple data modalities, broadening the types and complexity of data our models can utilize: for example, from plain text to image-caption pairs. Most multimodal learning algorithms focus on modeling simple one-to-one pairs of data from two modalities, such as image-caption pairs, or audio-text pairs. However, in most real-world settings, entities of different modalities interact with each other in more complex and multifaceted ways, going beyond one-to-one mappings. We propose to represent these complex relationships as graphs, allowing us to capture data with any number of modalities, and with complex relationships between modalities that can flexibly vary from one sample to another. Toward this goal, we propose Multimodal Graph Learning (MMGL), a general and systematic framework for capturing information from multiple multimodal neighbors with relational structures among them. In particular, we focus on MMGL for generative tasks, building upon pretrained Language Models (LMs), aiming to augment their text generation with multimodal neighbor contexts. We study three research questions raised by MMGL: (1) how can we infuse multiple neighbor information into the pretrained LMs, while avoiding scalability issues? (2) how can we infuse the graph structure information among multimodal neighbors into the LMs? and (3) how can we finetune the pretrained LMs to learn from the neighbor context in a parameter-efficient manner? We conduct extensive experiments to answer these three questions on MMGL and analyze the empirical results to pave the way for future MMGL research.

{{</citation>}}


### (109/185) An Ontology of Co-Creative AI Systems (Zhiyu Lin et al., 2023)

{{<citation>}}

Zhiyu Lin, Mark Riedl. (2023)  
**An Ontology of Co-Creative AI Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07472v1)  

---


**ABSTRACT**  
The term co-creativity has been used to describe a wide variety of human-AI assemblages in which human and AI are both involved in a creative endeavor. In order to assist with disambiguating research efforts, we present an ontology of co-creative systems, focusing on how responsibilities are divided between human and AI system and the information exchanged between them. We extend Lubart's original ontology of creativity support tools with three new categories emphasizing artificial intelligence: computer-as-subcontractor, computer-as-critic, and computer-as-teammate, some of which have sub-categorizations.

{{</citation>}}


### (110/185) What can knowledge graph alignment gain with Neuro-Symbolic learning approaches? (Pedro Giesteira Cotovio et al., 2023)

{{<citation>}}

Pedro Giesteira Cotovio, Ernesto Jimenez-Ruiz, Catia Pesquita. (2023)  
**What can knowledge graph alignment gain with Neuro-Symbolic learning approaches?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-SC, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.07417v1)  

---


**ABSTRACT**  
Knowledge Graphs (KG) are the backbone of many data-intensive applications since they can represent data coupled with its meaning and context. Aligning KGs across different domains and providers is necessary to afford a fuller and integrated representation. A severe limitation of current KG alignment (KGA) algorithms is that they fail to articulate logical thinking and reasoning with lexical, structural, and semantic data learning. Deep learning models are increasingly popular for KGA inspired by their good performance in other tasks, but they suffer from limitations in explainability, reasoning, and data efficiency. Hybrid neurosymbolic learning models hold the promise of integrating logical and data perspectives to produce high-quality alignments that are explainable and support validation through human-centric approaches. This paper examines the current state of the art in KGA and explores the potential for neurosymbolic integration, highlighting promising research directions for combining these fields.

{{</citation>}}


### (111/185) Give and Take: Federated Transfer Learning for Industrial IoT Network Intrusion Detection (Lochana Telugu Rajesh et al., 2023)

{{<citation>}}

Lochana Telugu Rajesh, Tapadhir Das, Raj Mani Shukla, Shamik Sengupta. (2023)  
**Give and Take: Federated Transfer Learning for Industrial IoT Network Intrusion Detection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2310.07354v1)  

---


**ABSTRACT**  
The rapid growth in Internet of Things (IoT) technology has become an integral part of today's industries forming the Industrial IoT (IIoT) initiative, where industries are leveraging IoT to improve communication and connectivity via emerging solutions like data analytics and cloud computing. Unfortunately, the rapid use of IoT has made it an attractive target for cybercriminals. Therefore, protecting these systems is of utmost importance. In this paper, we propose a federated transfer learning (FTL) approach to perform IIoT network intrusion detection. As part of the research, we also propose a combinational neural network as the centerpiece for performing FTL. The proposed technique splits IoT data between the client and server devices to generate corresponding models, and the weights of the client models are combined to update the server model. Results showcase high performance for the FTL setup between iterations on both the IIoT clients and the server. Additionally, the proposed FTL setup achieves better overall performance than contemporary machine learning algorithms at performing network intrusion detection.

{{</citation>}}


### (112/185) Semantic Association Rule Learning from Time Series Data and Knowledge Graphs (Erkan Karabulut et al., 2023)

{{<citation>}}

Erkan Karabulut, Victoria Degeler, Paul Groth. (2023)  
**Semantic Association Rule Learning from Time Series Data and Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph, Time Series  
[Paper Link](http://arxiv.org/abs/2310.07348v1)  

---


**ABSTRACT**  
Digital Twins (DT) are a promising concept in cyber-physical systems research due to their advanced features including monitoring and automated reasoning. Semantic technologies such as Knowledge Graphs (KG) are recently being utilized in DTs especially for information modelling. Building on this move, this paper proposes a pipeline for semantic association rule learning in DTs using KGs and time series data. In addition to this initial pipeline, we also propose new semantic association rule criterion. The approach is evaluated on an industrial water network scenario. Initial evaluation shows that the proposed approach is able to learn a high number of association rules with semantic information which are more generalizable. The paper aims to set a foundation for further work on using semantic association rule learning especially in the context of industrial applications.

{{</citation>}}


### (113/185) Beyond Memorization: Violating Privacy Via Inference with Large Language Models (Robin Staab et al., 2023)

{{<citation>}}

Robin Staab, Mark Vero, Mislav Balunović, Martin Vechev. (2023)  
**Beyond Memorization: Violating Privacy Via Inference with Large Language Models**  

---
Primary Category: cs.AI  
Categories: I-2-7, cs-AI, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07298v1)  

---


**ABSTRACT**  
Current privacy research on large language models (LLMs) primarily focuses on the issue of extracting memorized training data. At the same time, models' inference capabilities have increased drastically. This raises the key question of whether current LLMs could violate individuals' privacy by inferring personal attributes from text given at inference time. In this work, we present the first comprehensive study on the capabilities of pretrained LLMs to infer personal attributes from text. We construct a dataset consisting of real Reddit profiles, and show that current LLMs can infer a wide range of personal attributes (e.g., location, income, sex), achieving up to $85\%$ top-1 and $95.8\%$ top-3 accuracy at a fraction of the cost ($100\times$) and time ($240\times$) required by humans. As people increasingly interact with LLM-powered chatbots across all aspects of life, we also explore the emerging threat of privacy-invasive chatbots trying to extract personal information through seemingly benign questions. Finally, we show that common mitigations, i.e., text anonymization and model alignment, are currently ineffective at protecting user privacy against LLM inference. Our findings highlight that current LLMs can infer personal data at a previously unattainable scale. In the absence of working defenses, we advocate for a broader discussion around LLM privacy implications beyond memorization, striving for a wider privacy protection.

{{</citation>}}


### (114/185) An Analysis on Large Language Models in Healthcare: A Case Study of BioBERT (Shyni Sharaf et al., 2023)

{{<citation>}}

Shyni Sharaf, V. S. Anoop. (2023)  
**An Analysis on Large Language Models in Healthcare: A Case Study of BioBERT**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: BERT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.07282v2)  

---


**ABSTRACT**  
This paper conducts a comprehensive investigation into applying large language models, particularly on BioBERT, in healthcare. It begins with thoroughly examining previous natural language processing (NLP) approaches in healthcare, shedding light on the limitations and challenges these methods face. Following that, this research explores the path that led to the incorporation of BioBERT into healthcare applications, highlighting its suitability for addressing the specific requirements of tasks related to biomedical text mining. The analysis outlines a systematic methodology for fine-tuning BioBERT to meet the unique needs of the healthcare domain. This approach includes various components, including the gathering of data from a wide range of healthcare sources, data annotation for tasks like identifying medical entities and categorizing them, and the application of specialized preprocessing techniques tailored to handle the complexities found in biomedical texts. Additionally, the paper covers aspects related to model evaluation, with a focus on healthcare benchmarks and functions like processing of natural language in biomedical, question-answering, clinical document classification, and medical entity recognition. It explores techniques to improve the model's interpretability and validates its performance compared to existing healthcare-focused language models. The paper thoroughly examines ethical considerations, particularly patient privacy and data security. It highlights the benefits of incorporating BioBERT into healthcare contexts, including enhanced clinical decision support and more efficient information retrieval. Nevertheless, it acknowledges the impediments and complexities of this integration, encompassing concerns regarding data privacy, transparency, resource-intensive requirements, and the necessity for model customization to align with diverse healthcare domains.

{{</citation>}}


### (115/185) State of the Art on Diffusion Models for Visual Computing (Ryan Po et al., 2023)

{{<citation>}}

Ryan Po, Wang Yifan, Vladislav Golyanik, Kfir Aberman, Jonathan T. Barron, Amit H. Bermano, Eric Ryan Chan, Tali Dekel, Aleksander Holynski, Angjoo Kanazawa, C. Karen Liu, Lingjie Liu, Ben Mildenhall, Matthias Nießner, Björn Ommer, Christian Theobalt, Peter Wonka, Gordon Wetzstein. (2023)  
**State of the Art on Diffusion Models for Visual Computing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07204v1)  

---


**ABSTRACT**  
The field of visual computing is rapidly advancing due to the emergence of generative artificial intelligence (AI), which unlocks unprecedented capabilities for the generation, editing, and reconstruction of images, videos, and 3D scenes. In these domains, diffusion models are the generative AI architecture of choice. Within the last year alone, the literature on diffusion-based tools and applications has seen exponential growth and relevant papers are published across the computer graphics, computer vision, and AI communities with new works appearing daily on arXiv. This rapid growth of the field makes it difficult to keep up with all recent developments. The goal of this state-of-the-art report (STAR) is to introduce the basic mathematical concepts of diffusion models, implementation details and design choices of the popular Stable Diffusion model, as well as overview important aspects of these generative AI tools, including personalization, conditioning, inversion, among others. Moreover, we give a comprehensive overview of the rapidly growing literature on diffusion-based generation and editing, categorized by the type of generated medium, including 2D images, videos, 3D objects, locomotion, and 4D scenes. Finally, we discuss available datasets, metrics, open challenges, and social implications. This STAR provides an intuitive starting point to explore this exciting topic for researchers, artists, and practitioners alike.

{{</citation>}}


### (116/185) Leveraging Twitter Data for Sentiment Analysis of Transit User Feedback: An NLP Framework (Adway Das et al., 2023)

{{<citation>}}

Adway Das, Abhishek Kumar Prajapati, Pengxiang Zhang, Mukund Srinath, Andisheh Ranjbari. (2023)  
**Leveraging Twitter Data for Sentiment Analysis of Transit User Feedback: An NLP Framework**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SI, cs.AI  
Keywords: NLP, Sentiment Analysis, Twitter  
[Paper Link](http://arxiv.org/abs/2310.07086v1)  

---


**ABSTRACT**  
Traditional methods of collecting user feedback through transit surveys are often time-consuming, resource intensive, and costly. In this paper, we propose a novel NLP-based framework that harnesses the vast, abundant, and inexpensive data available on social media platforms like Twitter to understand users' perceptions of various service issues. Twitter, being a microblogging platform, hosts a wealth of real-time user-generated content that often includes valuable feedback and opinions on various products, services, and experiences. The proposed framework streamlines the process of gathering and analyzing user feedback without the need for costly and time-consuming user feedback surveys using two techniques. First, it utilizes few-shot learning for tweet classification within predefined categories, allowing effective identification of the issues described in tweets. It then employs a lexicon-based sentiment analysis model to assess the intensity and polarity of the tweet sentiments, distinguishing between positive, negative, and neutral tweets. The effectiveness of the framework was validated on a subset of manually labeled Twitter data and was applied to the NYC subway system as a case study. The framework accurately classifies tweets into predefined categories related to safety, reliability, and maintenance of the subway system and effectively measured sentiment intensities within each category. The general findings were corroborated through a comparison with an agency-run customer survey conducted in the same year. The findings highlight the effectiveness of the proposed framework in gauging user feedback through inexpensive social media data to understand the pain points of the transit system and plan for targeted improvements.

{{</citation>}}


## cs.CV (31)



### (117/185) LangNav: Language as a Perceptual Representation for Navigation (Bowen Pan et al., 2023)

{{<citation>}}

Bowen Pan, Rameswar Panda, SouYoung Jin, Rogerio Feris, Aude Oliva, Phillip Isola, Yoon Kim. (2023)  
**LangNav: Language as a Perceptual Representation for Navigation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.07889v1)  

---


**ABSTRACT**  
We explore the use of language as a perceptual representation for vision-and-language navigation. Our approach uses off-the-shelf vision systems (for image captioning and object detection) to convert an agent's egocentric panoramic view at each time step into natural language descriptions. We then finetune a pretrained language model to select an action, based on the current view and the trajectory history, that would best fulfill the navigation instructions. In contrast to the standard setup which adapts a pretrained language model to work directly with continuous visual features from pretrained vision models, our approach instead uses (discrete) language as the perceptual representation. We explore two use cases of our language-based navigation (LangNav) approach on the R2R vision-and-language navigation benchmark: generating synthetic trajectories from a prompted large language model (GPT-4) with which to finetune a smaller language model; and sim-to-real transfer where we transfer a policy learned on a simulated environment (ALFRED) to a real-world environment (R2R). Our approach is found to improve upon strong baselines that rely on visual features in settings where only a few gold trajectories (10-100) are available, demonstrating the potential of using language as a perceptual representation for navigation tasks.

{{</citation>}}


### (118/185) CrIBo: Self-Supervised Learning via Cross-Image Object-Level Bootstrapping (Tim Lebailly et al., 2023)

{{<citation>}}

Tim Lebailly, Thomas Stegmüller, Behzad Bozorgtabar, Jean-Philippe Thiran, Tinne Tuytelaars. (2023)  
**CrIBo: Self-Supervised Learning via Cross-Image Object-Level Bootstrapping**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.07855v1)  

---


**ABSTRACT**  
Leveraging nearest neighbor retrieval for self-supervised representation learning has proven beneficial with object-centric images. However, this approach faces limitations when applied to scene-centric datasets, where multiple objects within an image are only implicitly captured in the global representation. Such global bootstrapping can lead to undesirable entanglement of object representations. Furthermore, even object-centric datasets stand to benefit from a finer-grained bootstrapping approach. In response to these challenges, we introduce a novel Cross-Image Object-Level Bootstrapping method tailored to enhance dense visual representation learning. By employing object-level nearest neighbor bootstrapping throughout the training, CrIBo emerges as a notably strong and adequate candidate for in-context learning, leveraging nearest neighbor retrieval at test time. CrIBo shows state-of-the-art performance on the latter task while being highly competitive in more standard downstream segmentation tasks. Our code and pretrained models will be publicly available upon acceptance.

{{</citation>}}


### (119/185) 3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers (Jieneng Chen et al., 2023)

{{<citation>}}

Jieneng Chen, Jieru Mei, Xianhang Li, Yongyi Lu, Qihang Yu, Qingyue Wei, Xiangde Luo, Yutong Xie, Ehsan Adeli, Yan Wang, Matthew Lungren, Lei Xing, Le Lu, Alan Yuille, Yuyin Zhou. (2023)  
**3D TransUNet: Advancing Medical Image Segmentation through Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07781v1)  

---


**ABSTRACT**  
Medical image segmentation plays a crucial role in advancing healthcare systems for disease diagnosis and treatment planning. The u-shaped architecture, popularly known as U-Net, has proven highly successful for various medical image segmentation tasks. However, U-Net's convolution-based operations inherently limit its ability to model long-range dependencies effectively. To address these limitations, researchers have turned to Transformers, renowned for their global self-attention mechanisms, as alternative architectures. One popular network is our previous TransUNet, which leverages Transformers' self-attention to complement U-Net's localized information with the global context. In this paper, we extend the 2D TransUNet architecture to a 3D network by building upon the state-of-the-art nnU-Net architecture, and fully exploring Transformers' potential in both the encoder and decoder design. We introduce two key components: 1) A Transformer encoder that tokenizes image patches from a convolution neural network (CNN) feature map, enabling the extraction of global contexts, and 2) A Transformer decoder that adaptively refines candidate regions by utilizing cross-attention between candidate proposals and U-Net features. Our investigations reveal that different medical tasks benefit from distinct architectural designs. The Transformer encoder excels in multi-organ segmentation, where the relationship among organs is crucial. On the other hand, the Transformer decoder proves more beneficial for dealing with small and challenging segmented targets such as tumor segmentation. Extensive experiments showcase the significant potential of integrating a Transformer-based encoder and decoder into the u-shaped medical image segmentation architecture. TransUNet outperforms competitors in various medical applications.

{{</citation>}}


### (120/185) PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection (Qiang Zhou et al., 2023)

{{<citation>}}

Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, Hao Zhao. (2023)  
**PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.07716v1)  

---


**ABSTRACT**  
Object anomaly detection is an important problem in the field of machine vision and has seen remarkable progress recently. However, two significant challenges hinder its research and application. First, existing datasets lack comprehensive visual information from various pose angles. They usually have an unrealistic assumption that the anomaly-free training dataset is pose-aligned, and the testing samples have the same pose as the training data. However, in practice, anomaly may exist in any regions on a object, the training and query samples may have different poses, calling for the study on pose-agnostic anomaly detection. Second, the absence of a consensus on experimental protocols for pose-agnostic anomaly detection leads to unfair comparisons of different methods, hindering the research on pose-agnostic anomaly detection. To address these issues, we develop Multi-pose Anomaly Detection (MAD) dataset and Pose-agnostic Anomaly Detection (PAD) benchmark, which takes the first step to address the pose-agnostic anomaly detection problem. Specifically, we build MAD using 20 complex-shaped LEGO toys including 4K views with various poses, and high-quality and diverse 3D anomalies in both simulated and real environments. Additionally, we propose a novel method OmniposeAD, trained using MAD, specifically designed for pose-agnostic anomaly detection. Through comprehensive evaluations, we demonstrate the relevance of our dataset and method. Furthermore, we provide an open-source benchmark library, including dataset and baseline methods that cover 8 anomaly detection paradigms, to facilitate future research and application in this domain. Code, data, and models are publicly available at https://github.com/EricLee0224/PAD.

{{</citation>}}


### (121/185) OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation (Jie An et al., 2023)

{{<citation>}}

Jie An, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Lijuan Wang, Jiebo Luo. (2023)  
**OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2310.07749v1)  

---


**ABSTRACT**  
This work investigates a challenging task named open-domain interleaved image-text generation, which generates interleaved texts and images following an input query. We propose a new interleaved generation framework based on prompting large-language models (LLMs) and pre-trained text-to-image (T2I) models, namely OpenLEAF. In OpenLEAF, the LLM generates textual descriptions, coordinates T2I models, creates visual prompts for generating images, and incorporates global contexts into the T2I models. This global context improves the entity and style consistencies of images in the interleaved generation. For model assessment, we first propose to use large multi-modal models (LMMs) to evaluate the entity and style consistencies of open-domain interleaved image-text sequences. According to the LMM evaluation on our constructed evaluation set, the proposed interleaved generation framework can generate high-quality image-text content for various domains and applications, such as how-to question answering, storytelling, graphical story rewriting, and webpage/poster generation tasks. Moreover, we validate the effectiveness of the proposed LMM evaluation technique with human assessment. We hope our proposed framework, benchmark, and LMM evaluation could help establish the intriguing interleaved image-text generation task.

{{</citation>}}


### (122/185) Ferret: Refer and Ground Anything Anywhere at Any Granularity (Haoxuan You et al., 2023)

{{<citation>}}

Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, Yinfei Yang. (2023)  
**Ferret: Refer and Ground Anything Anywhere at Any Granularity**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07704v1)  

---


**ABSTRACT**  
We introduce Ferret, a new Multimodal Large Language Model (MLLM) capable of understanding spatial referring of any shape or granularity within an image and accurately grounding open-vocabulary descriptions. To unify referring and grounding in the LLM paradigm, Ferret employs a novel and powerful hybrid region representation that integrates discrete coordinates and continuous features jointly to represent a region in the image. To extract the continuous features of versatile regions, we propose a spatial-aware visual sampler, adept at handling varying sparsity across different shapes. Consequently, Ferret can accept diverse region inputs, such as points, bounding boxes, and free-form shapes. To bolster the desired capability of Ferret, we curate GRIT, a comprehensive refer-and-ground instruction tuning dataset including 1.1M samples that contain rich hierarchical spatial knowledge, with 95K hard negative data to promote model robustness. The resulting model not only achieves superior performance in classical referring and grounding tasks, but also greatly outperforms existing MLLMs in region-based and localization-demanded multimodal chatting. Our evaluations also reveal a significantly improved capability of describing image details and a remarkable alleviation in object hallucination. Code and data will be available at https://github.com/apple/ml-ferret

{{</citation>}}


### (123/185) HaarNet: Large-scale Linear-Morphological Hybrid Network for RGB-D Semantic Segmentation (Rick Groenendijk et al., 2023)

{{<citation>}}

Rick Groenendijk, Leo Dorst, Theo Gevers. (2023)  
**HaarNet: Large-scale Linear-Morphological Hybrid Network for RGB-D Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: I-4-6; I-2-6, cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.07669v1)  

---


**ABSTRACT**  
Signals from different modalities each have their own combination algebra which affects their sampling processing. RGB is mostly linear; depth is a geometric signal following the operations of mathematical morphology. If a network obtaining RGB-D input has both kinds of operators available in its layers, it should be able to give effective output with fewer parameters. In this paper, morphological elements in conjunction with more familiar linear modules are used to construct a mixed linear-morphological network called HaarNet. This is the first large-scale linear-morphological hybrid, evaluated on a set of sizeable real-world datasets. In the network, morphological Haar sampling is applied to both feature channels in several layers, which splits extreme values and high-frequency information such that both can be processed to improve both modalities. Moreover, morphologically parameterised ReLU is used, and morphologically-sound up-sampling is applied to obtain a full-resolution output. Experiments show that HaarNet is competitive with a state-of-the-art CNN, implying that morphological networks are a promising research direction for geometry-based learning tasks.

{{</citation>}}


### (124/185) Accelerating Vision Transformers Based on Heterogeneous Attention Patterns (Deli Yu et al., 2023)

{{<citation>}}

Deli Yu, Teng Xi, Jianwei Li, Baopu Li, Gang Zhang, Haocheng Feng, Junyu Han, Jingtuo Liu, Errui Ding, Jingdong Wang. (2023)  
**Accelerating Vision Transformers Based on Heterogeneous Attention Patterns**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07664v1)  

---


**ABSTRACT**  
Recently, Vision Transformers (ViTs) have attracted a lot of attention in the field of computer vision. Generally, the powerful representative capacity of ViTs mainly benefits from the self-attention mechanism, which has a high computation complexity. To accelerate ViTs, we propose an integrated compression pipeline based on observed heterogeneous attention patterns across layers. On one hand, different images share more similar attention patterns in early layers than later layers, indicating that the dynamic query-by-key self-attention matrix may be replaced with a static self-attention matrix in early layers. Then, we propose a dynamic-guided static self-attention (DGSSA) method where the matrix inherits self-attention information from the replaced dynamic self-attention to effectively improve the feature representation ability of ViTs. On the other hand, the attention maps have more low-rank patterns, which reflect token redundancy, in later layers than early layers. In a view of linear dimension reduction, we further propose a method of global aggregation pyramid (GLAD) to reduce the number of tokens in later layers of ViTs, such as Deit. Experimentally, the integrated compression pipeline of DGSSA and GLAD can accelerate up to 121% run-time throughput compared with DeiT, which surpasses all SOTA approaches.

{{</citation>}}


### (125/185) A Discrepancy Aware Framework for Robust Anomaly Detection (Yuxuan Cai et al., 2023)

{{<citation>}}

Yuxuan Cai, Dingkang Liang, Dongliang Luo, Xinwei He, Xin Yang, Xiang Bai. (2023)  
**A Discrepancy Aware Framework for Robust Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.07585v1)  

---


**ABSTRACT**  
Defect detection is a critical research area in artificial intelligence. Recently, synthetic data-based self-supervised learning has shown great potential on this task. Although many sophisticated synthesizing strategies exist, little research has been done to investigate the robustness of models when faced with different strategies. In this paper, we focus on this issue and find that existing methods are highly sensitive to them. To alleviate this issue, we present a Discrepancy Aware Framework (DAF), which demonstrates robust performance consistently with simple and cheap strategies across different anomaly detection benchmarks. We hypothesize that the high sensitivity to synthetic data of existing self-supervised methods arises from their heavy reliance on the visual appearance of synthetic data during decoding. In contrast, our method leverages an appearance-agnostic cue to guide the decoder in identifying defects, thereby alleviating its reliance on synthetic appearance. To this end, inspired by existing knowledge distillation methods, we employ a teacher-student network, which is trained based on synthesized outliers, to compute the discrepancy map as the cue. Extensive experiments on two challenging datasets prove the robustness of our method. Under the simple synthesis strategies, it outperforms existing methods by a large margin. Furthermore, it also achieves the state-of-the-art localization performance. Code is available at: https://github.com/caiyuxuan1120/DAF.

{{</citation>}}


### (126/185) Relational Prior Knowledge Graphs for Detection and Instance Segmentation (Osman Ülger et al., 2023)

{{<citation>}}

Osman Ülger, Yu Wang, Ysbrand Galama, Sezer Karaoglu, Theo Gevers, Martin R. Oswald. (2023)  
**Relational Prior Knowledge Graphs for Detection and Instance Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.07573v1)  

---


**ABSTRACT**  
Humans have a remarkable ability to perceive and reason about the world around them by understanding the relationships between objects. In this paper, we investigate the effectiveness of using such relationships for object detection and instance segmentation. To this end, we propose a Relational Prior-based Feature Enhancement Model (RP-FEM), a graph transformer that enhances object proposal features using relational priors. The proposed architecture operates on top of scene graphs obtained from initial proposals and aims to concurrently learn relational context modeling for object detection and instance segmentation. Experimental evaluations on COCO show that the utilization of scene graphs, augmented with relational priors, offer benefits for object detection and instance segmentation. RP-FEM demonstrates its capacity to suppress improbable class predictions within the image while also preventing the model from generating duplicate predictions, leading to improvements over the baseline model on which it is built.

{{</citation>}}


### (127/185) Does resistance to Style-Transfer equal Shape Bias? Evaluating Shape Bias by Distorted Shape (Ziqi Wen et al., 2023)

{{<citation>}}

Ziqi Wen, Tianqin Li, Tai Sing Lee. (2023)  
**Does resistance to Style-Transfer equal Shape Bias? Evaluating Shape Bias by Distorted Shape**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.07555v1)  

---


**ABSTRACT**  
Deep learning models are known to exhibit a strong texture bias, while human tends to rely heavily on global shape for object recognition. The current benchmark for evaluating a model's shape bias is a set of style-transferred images with the assumption that resistance to the attack of style transfer is related to the development of shape sensitivity in the model. In this work, we show that networks trained with style-transfer images indeed learn to ignore style, but its shape bias arises primarily from local shapes. We provide a Distorted Shape Testbench (DiST) as an alternative measurement of global shape sensitivity. Our test includes 2400 original images from ImageNet-1K, each of which is accompanied by two images with the global shapes of the original image distorted while preserving its texture via the texture synthesis program. We found that (1) models that performed well on the previous shape bias evaluation do not fare well in the proposed DiST; (2) the widely adopted ViT models do not show significant advantages over Convolutional Neural Networks (CNNs) on this benchmark despite that ViTs rank higher on the previous shape bias tests. (3) training with DiST images bridges the significant gap between human and existing SOTA models' performance while preserving the models' accuracy on standard image classification tasks; training with DiST images and style-transferred images are complementary, and can be combined to train network together to enhance both the global and local shape sensitivity of the network. Our code will be host at: https://github.com/leelabcnbc/DiST

{{</citation>}}


### (128/185) ProtoHPE: Prototype-guided High-frequency Patch Enhancement for Visible-Infrared Person Re-identification (Guiwei Zhang et al., 2023)

{{<citation>}}

Guiwei Zhang, Yongfei Zhang, Zichang Tan. (2023)  
**ProtoHPE: Prototype-guided High-frequency Patch Enhancement for Visible-Infrared Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07552v1)  

---


**ABSTRACT**  
Visible-infrared person re-identification is challenging due to the large modality gap. To bridge the gap, most studies heavily rely on the correlation of visible-infrared holistic person images, which may perform poorly under severe distribution shifts. In contrast, we find that some cross-modal correlated high-frequency components contain discriminative visual patterns and are less affected by variations such as wavelength, pose, and background clutter than holistic images. Therefore, we are motivated to bridge the modality gap based on such high-frequency components, and propose \textbf{Proto}type-guided \textbf{H}igh-frequency \textbf{P}atch \textbf{E}nhancement (ProtoHPE) with two core designs. \textbf{First}, to enhance the representation ability of cross-modal correlated high-frequency components, we split patches with such components by Wavelet Transform and exponential moving average Vision Transformer (ViT), then empower ViT to take the split patches as auxiliary input. \textbf{Second}, to obtain semantically compact and discriminative high-frequency representations of the same identity, we propose Multimodal Prototypical Contrast. To be specific, it hierarchically captures the comprehensive semantics of different modal instances, facilitating the aggregation of high-frequency representations belonging to the same identity. With it, ViT can capture key high-frequency components during inference without relying on ProtoHPE, thus bringing no extra complexity. Extensive experiments validate the effectiveness of ProtoHPE.

{{</citation>}}


### (129/185) Attribute Localization and Revision Network for Zero-Shot Learning (Junzhe Xu et al., 2023)

{{<citation>}}

Junzhe Xu, Suling Duan, Chenwei Tang, Zhenan He, Jiancheng Lv. (2023)  
**Attribute Localization and Revision Network for Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.07548v1)  

---


**ABSTRACT**  
Zero-shot learning enables the model to recognize unseen categories with the aid of auxiliary semantic information such as attributes. Current works proposed to detect attributes from local image regions and align extracted features with class-level semantics. In this paper, we find that the choice between local and global features is not a zero-sum game, global features can also contribute to the understanding of attributes. In addition, aligning attribute features with class-level semantics ignores potential intra-class attribute variation. To mitigate these disadvantages, we present Attribute Localization and Revision Network in this paper. First, we design Attribute Localization Module (ALM) to capture both local and global features from image regions, a novel module called Scale Control Unit is incorporated to fuse global and local representations. Second, we propose Attribute Revision Module (ARM), which generates image-level semantics by revising the ground-truth value of each attribute, compensating for performance degradation caused by ignoring intra-class variation. Finally, the output of ALM will be aligned with revised semantics produced by ARM to achieve the training process. Comprehensive experimental results on three widely used benchmarks demonstrate the effectiveness of our model in the zero-shot prediction task.

{{</citation>}}


### (130/185) S4C: Self-Supervised Semantic Scene Completion with Neural Fields (Adrian Hayler et al., 2023)

{{<citation>}}

Adrian Hayler, Felix Wimbauer, Dominik Muhle, Christian Rupprecht, Daniel Cremers. (2023)  
**S4C: Self-Supervised Semantic Scene Completion with Neural Fields**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.07522v2)  

---


**ABSTRACT**  
3D semantic scene understanding is a fundamental challenge in computer vision. It enables mobile agents to autonomously plan and navigate arbitrary environments. SSC formalizes this challenge as jointly estimating dense geometry and semantic information from sparse observations of a scene. Current methods for SSC are generally trained on 3D ground truth based on aggregated LiDAR scans. This process relies on special sensors and annotation by hand which are costly and do not scale well. To overcome this issue, our work presents the first self-supervised approach to SSC called S4C that does not rely on 3D ground truth data. Our proposed method can reconstruct a scene from a single image and only relies on videos and pseudo segmentation ground truth generated from off-the-shelf image segmentation network during training. Unlike existing methods, which use discrete voxel grids, we represent scenes as implicit semantic fields. This formulation allows querying any point within the camera frustum for occupancy and semantic class. Our architecture is trained through rendering-based self-supervised losses. Nonetheless, our method achieves performance close to fully supervised state-of-the-art methods. Additionally, our method demonstrates strong generalization capabilities and can synthesize accurate segmentation maps for far away viewpoints.

{{</citation>}}


### (131/185) Heuristic Vision Pre-Training with Self-Supervised and Supervised Multi-Task Learning (Zhiming Qian, 2023)

{{<citation>}}

Zhiming Qian. (2023)  
**Heuristic Vision Pre-Training with Self-Supervised and Supervised Multi-Task Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.07510v1)  

---


**ABSTRACT**  
To mimic human vision with the way of recognizing the diverse and open world, foundation vision models are much critical. While recent techniques of self-supervised learning show the promising potentiality of this mission, we argue that signals from labelled data are also important for common-sense recognition, and properly chosen pre-text tasks can facilitate the efficiency of vision representation learning. To this end, we propose a novel pre-training framework by adopting both self-supervised and supervised visual pre-text tasks in a multi-task manner. Specifically, given an image, we take a heuristic way by considering its intrinsic style properties, inside objects with their locations and correlations, and how it looks like in 3D space for basic visual understanding. However, large-scale object bounding boxes and correlations are usually hard to achieve. Alternatively, we develop a hybrid method by leveraging both multi-label classification and self-supervised learning. On the one hand, under the multi-label supervision, the pre-trained model can explore the detailed information of an image, e.g., image types, objects, and part of semantic relations. On the other hand, self-supervised learning tasks, with respect to Masked Image Modeling (MIM) and contrastive learning, can help the model learn pixel details and patch correlations. Results show that our pre-trained models can deliver results on par with or better than state-of-the-art (SOTA) results on multiple visual tasks. For example, with a vanilla Swin-B backbone, we achieve 85.3\% top-1 accuracy on ImageNet-1K classification, 47.9 box AP on COCO object detection for Mask R-CNN, and 50.6 mIoU on ADE-20K semantic segmentation when using Upernet. The performance shows the ability of our vision foundation model to serve general purpose vision tasks.

{{</citation>}}


### (132/185) Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation (Haizhong Zheng et al., 2023)

{{<citation>}}

Haizhong Zheng, Jiachen Sun, Shutong Wu, Bhavya Kailkhura, Zhuoqing Mao, Chaowei Xiao, Atul Prakash. (2023)  
**Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.07506v1)  

---


**ABSTRACT**  
Given a real-world dataset, data condensation (DC) aims to synthesize a significantly smaller dataset that captures the knowledge of this dataset for model training with high performance. Recent works propose to enhance DC with data parameterization, which condenses data into parameterized data containers rather than pixel space. The intuition behind data parameterization is to encode shared features of images to avoid additional storage costs. In this paper, we recognize that images share common features in a hierarchical way due to the inherent hierarchical structure of the classification system, which is overlooked by current data parameterization methods. To better align DC with this hierarchical nature and encourage more efficient information sharing inside data containers, we propose a novel data parameterization architecture, Hierarchical Memory Network (HMN). HMN stores condensed data in a three-tier structure, representing the dataset-level, class-level, and instance-level features. Another helpful property of the hierarchical architecture is that HMN naturally ensures good independence among images despite achieving information sharing. This enables instance-level pruning for HMN to reduce redundant information, thereby further minimizing redundancy and enhancing performance. We evaluate HMN on four public datasets (SVHN, CIFAR10, CIFAR100, and Tiny-ImageNet) and compare HMN with eight DC baselines. The evaluation results show that our proposed method outperforms all baselines, even when trained with a batch-based loss consuming less GPU memory.

{{</citation>}}


### (133/185) Distance-based Weighted Transformer Network for Image Completion (Pourya Shamsolmoali et al., 2023)

{{<citation>}}

Pourya Shamsolmoali, Masoumeh Zareapoor, Huiyu Zhou, Xuelong Li, Yue Lu. (2023)  
**Distance-based Weighted Transformer Network for Image Completion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07440v1)  

---


**ABSTRACT**  
The challenge of image generation has been effectively modeled as a problem of structure priors or transformation. However, existing models have unsatisfactory performance in understanding the global input image structures because of particular inherent features (for example, local inductive prior). Recent studies have shown that self-attention is an efficient modeling technique for image completion problems. In this paper, we propose a new architecture that relies on Distance-based Weighted Transformer (DWT) to better understand the relationships between an image's components. In our model, we leverage the strengths of both Convolutional Neural Networks (CNNs) and DWT blocks to enhance the image completion process. Specifically, CNNs are used to augment the local texture information of coarse priors and DWT blocks are used to recover certain coarse textures and coherent visual structures. Unlike current approaches that generally use CNNs to create feature maps, we use the DWT to encode global dependencies and compute distance-based weighted feature maps, which substantially minimizes the problem of visual ambiguities. Meanwhile, to better produce repeated textures, we introduce Residual Fast Fourier Convolution (Res-FFC) blocks to combine the encoder's skip features with the coarse features provided by our generator. Furthermore, a simple yet effective technique is proposed to normalize the non-zero values of convolutions, and fine-tune the network layers for regularization of the gradient norms to provide an efficient training stabiliser. Extensive quantitative and qualitative experiments on three challenging datasets demonstrate the superiority of our proposed model compared to existing approaches.

{{</citation>}}


### (134/185) Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else (Hazarapet Tunanyan et al., 2023)

{{<citation>}}

Hazarapet Tunanyan, Dejia Xu, Shant Navasardyan, Zhangyang Wang, Humphrey Shi. (2023)  
**Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.07419v1)  

---


**ABSTRACT**  
Recent advances in text-to-image diffusion models have enabled the photorealistic generation of images from text prompts. Despite the great progress, existing models still struggle to generate compositional multi-concept images naturally, limiting their ability to visualize human imagination. While several recent works have attempted to address this issue, they either introduce additional training or adopt guidance at inference time. In this work, we consider a more ambitious goal: natural multi-concept generation using a pre-trained diffusion model, and with almost no extra cost. To achieve this goal, we identify the limitations in the text embeddings used for the pre-trained text-to-image diffusion models. Specifically, we observe concept dominance and non-localized contribution that severely degrade multi-concept generation performance. We further design a minimal low-cost solution that overcomes the above issues by tweaking (not re-training) the text embeddings for more realistic multi-concept text-to-image generation. Our Correction by Similarities method tweaks the embedding of concepts by collecting semantic features from most similar tokens to localize the contribution. To avoid mixing features of concepts, we also apply Cross-Token Non-Maximum Suppression, which excludes the overlap of contributions from different concepts. Experiments show that our approach outperforms previous methods in text-to-image, image manipulation, and personalization tasks, despite not introducing additional training or inference costs to the diffusion steps.

{{</citation>}}


### (135/185) CLIP for Lightweight Semantic Segmentation (Ke Jin et al., 2023)

{{<citation>}}

Ke Jin, Wankou Yang. (2023)  
**CLIP for Lightweight Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.07394v1)  

---


**ABSTRACT**  
The large-scale pretrained model CLIP, trained on 400 million image-text pairs, offers a promising paradigm for tackling vision tasks, albeit at the image level. Later works, such as DenseCLIP and LSeg, extend this paradigm to dense prediction, including semantic segmentation, and have achieved excellent results. However, the above methods either rely on CLIP-pretrained visual backbones or use none-pretrained but heavy backbones such as Swin, while falling ineffective when applied to lightweight backbones. The reason for this is that the lightweitht networks, feature extraction ability of which are relatively limited, meet difficulty embedding the image feature aligned with text embeddings perfectly. In this work, we present a new feature fusion module which tackles this problem and enables language-guided paradigm to be applied to lightweight networks. Specifically, the module is a parallel design of CNN and transformer with a two-way bridge in between, where CNN extracts spatial information and visual context of the feature map from the image encoder, and the transformer propagates text embeddings from the text encoder forward. The core of the module is the bidirectional fusion of visual and text feature across the bridge which prompts their proximity and alignment in embedding space. The module is model-agnostic, which can not only make language-guided lightweight semantic segmentation practical, but also fully exploit the pretrained knowledge of language priors and achieve better performance than previous SOTA work, such as DenseCLIP, whatever the vision backbone is. Extensive experiments have been conducted to demonstrate the superiority of our method.

{{</citation>}}


### (136/185) Causal Unsupervised Semantic Segmentation (Junho Kim et al., 2023)

{{<citation>}}

Junho Kim, Byung-Kwan Lee, Yong Man Ro. (2023)  
**Causal Unsupervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.07379v1)  

---


**ABSTRACT**  
Unsupervised semantic segmentation aims to achieve high-quality semantic grouping without human-labeled annotations. With the advent of self-supervised pre-training, various frameworks utilize the pre-trained features to train prediction heads for unsupervised dense prediction. However, a significant challenge in this unsupervised setup is determining the appropriate level of clustering required for segmenting concepts. To address it, we propose a novel framework, CAusal Unsupervised Semantic sEgmentation (CAUSE), which leverages insights from causal inference. Specifically, we bridge intervention-oriented approach (i.e., frontdoor adjustment) to define suitable two-step tasks for unsupervised prediction. The first step involves constructing a concept clusterbook as a mediator, which represents possible concept prototypes at different levels of granularity in a discretized form. Then, the mediator establishes an explicit link to the subsequent concept-wise self-supervised learning for pixel-level grouping. Through extensive experiments and analyses on various datasets, we corroborate the effectiveness of CAUSE and achieve state-of-the-art performance in unsupervised semantic segmentation.

{{</citation>}}


### (137/185) Domain Generalization Guided by Gradient Signal to Noise Ratio of Parameters (Mateusz Michalkiewicz et al., 2023)

{{<citation>}}

Mateusz Michalkiewicz, Masoud Faraki, Xiang Yu, Manmohan Chandraker, Mahsa Baktashmotlagh. (2023)  
**Domain Generalization Guided by Gradient Signal to Noise Ratio of Parameters**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.07361v1)  

---


**ABSTRACT**  
Overfitting to the source domain is a common issue in gradient-based training of deep neural networks. To compensate for the over-parameterized models, numerous regularization techniques have been introduced such as those based on dropout. While these methods achieve significant improvements on classical benchmarks such as ImageNet, their performance diminishes with the introduction of domain shift in the test set i.e. when the unseen data comes from a significantly different distribution. In this paper, we move away from the classical approach of Bernoulli sampled dropout mask construction and propose to base the selection on gradient-signal-to-noise ratio (GSNR) of network's parameters. Specifically, at each training step, parameters with high GSNR will be discarded. Furthermore, we alleviate the burden of manually searching for the optimal dropout ratio by leveraging a meta-learning approach. We evaluate our method on standard domain generalization benchmarks and achieve competitive results on classification and face anti-spoofing problems.

{{</citation>}}


### (138/185) IMITATE: Clinical Prior Guided Hierarchical Vision-Language Pre-training (Che Liu et al., 2023)

{{<citation>}}

Che Liu, Sibo Cheng, Miaojing Shi, Anand Shah, Wenjia Bai, Rossella Arcucci. (2023)  
**IMITATE: Clinical Prior Guided Hierarchical Vision-Language Pre-training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2310.07355v1)  

---


**ABSTRACT**  
In the field of medical Vision-Language Pre-training (VLP), significant efforts have been devoted to deriving text and image features from both clinical reports and associated medical images. However, most existing methods may have overlooked the opportunity in leveraging the inherent hierarchical structure of clinical reports, which are generally split into `findings' for descriptive content and `impressions' for conclusive observation. Instead of utilizing this rich, structured format, current medical VLP approaches often simplify the report into either a unified entity or fragmented tokens. In this work, we propose a novel clinical prior guided VLP framework named IMITATE to learn the structure information from medical reports with hierarchical vision-language alignment. The framework derives multi-level visual features from the chest X-ray (CXR) images and separately aligns these features with the descriptive and the conclusive text encoded in the hierarchical medical report. Furthermore, a new clinical-informed contrastive loss is introduced for cross-modal learning, which accounts for clinical prior knowledge in formulating sample correlations in contrastive learning. The proposed model, IMITATE, outperforms baseline VLP methods across six different datasets, spanning five medical imaging downstream tasks. Comprehensive experimental results highlight the advantages of integrating the hierarchical structure of medical reports for vision-language alignment.

{{</citation>}}


### (139/185) Guided Attention for Interpretable Motion Captioning (Karim Radouane et al., 2023)

{{<citation>}}

Karim Radouane, Andon Tchechmedjiev, Sylvie Ranwez, Julien Lagarde. (2023)  
**Guided Attention for Interpretable Motion Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, BLEU  
[Paper Link](http://arxiv.org/abs/2310.07324v1)  

---


**ABSTRACT**  
While much effort has been invested in generating human motion from text, relatively few studies have been dedicated to the reverse direction, that is, generating text from motion. Much of the research focuses on maximizing generation quality without any regard for the interpretability of the architectures, particularly regarding the influence of particular body parts in the generation and the temporal synchronization of words with specific movements and actions. This study explores the combination of movement encoders with spatio-temporal attention models and proposes strategies to guide the attention during training to highlight perceptually pertinent areas of the skeleton in time. We show that adding guided attention with adaptive gate leads to interpretable captioning while improving performance compared to higher parameter-count non-interpretable SOTA systems. On the KIT MLD dataset, we obtain a BLEU@4 of 24.4% (SOTA+6%), a ROUGE-L of 58.30% (SOTA +14.1%), a CIDEr of 112.10 (SOTA +32.6) and a Bertscore of 41.20% (SOTA +18.20%). On HumanML3D, we obtain a BLEU@4 of 25.00 (SOTA +2.7%), a ROUGE-L score of 55.4% (SOTA +6.1%), a CIDEr of 61.6 (SOTA -10.9%), a Bertscore of 40.3% (SOTA +2.5%). Our code implementation and reproduction details will be soon available at https://github.com/rd20karim/M2T-Interpretable/tree/main.

{{</citation>}}


### (140/185) Deep Aramaic: Towards a Synthetic Data Paradigm Enabling Machine Learning in Epigraphy (Andrei C. Aioanei et al., 2023)

{{<citation>}}

Andrei C. Aioanei, Regine Hunziker-Rodewald, Konstantin Klein, Dominik L. Michels. (2023)  
**Deep Aramaic: Towards a Synthetic Data Paradigm Enabling Machine Learning in Epigraphy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07310v1)  

---


**ABSTRACT**  
Epigraphy increasingly turns to modern artificial intelligence (AI) technologies such as machine learning (ML) for extracting insights from ancient inscriptions. However, scarce labeled data for training ML algorithms severely limits current techniques, especially for ancient scripts like Old Aramaic. Our research pioneers an innovative methodology for generating synthetic training data tailored to Old Aramaic letters. Our pipeline synthesizes photo-realistic Aramaic letter datasets, incorporating textural features, lighting, damage, and augmentations to mimic real-world inscription diversity. Despite minimal real examples, we engineer a dataset of 250,000 training and 25,000 validation images covering the 22 letter classes in the Aramaic alphabet. This comprehensive corpus provides a robust volume of data for training a residual neural network (ResNet) to classify highly degraded Aramaic letters. The ResNet model demonstrates high accuracy in classifying real images from the 8th century BCE Hadad statue inscription. Additional experiments validate performance on varying materials and styles, proving effective generalization. Our results validate the model's capabilities in handling diverse real-world scenarios, proving the viability of our synthetic data approach and avoiding the dependence on scarce training data that has constrained epigraphic analysis. Our innovative framework elevates interpretation accuracy on damaged inscriptions, thus enhancing knowledge extraction from these historical resources.

{{</citation>}}


### (141/185) Distilling Efficient Vision Transformers from CNNs for Semantic Segmentation (Xu Zheng et al., 2023)

{{<citation>}}

Xu Zheng, Yunhao Luo, Pengyuan Zhou, Lin Wang. (2023)  
**Distilling Efficient Vision Transformers from CNNs for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, Semantic Segmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07265v1)  

---


**ABSTRACT**  
In this paper, we tackle a new problem: how to transfer knowledge from the pre-trained cumbersome yet well-performed CNN-based model to learn a compact Vision Transformer (ViT)-based model while maintaining its learning capacity? Due to the completely different characteristics of ViT and CNN and the long-existing capacity gap between teacher and student models in Knowledge Distillation (KD), directly transferring the cross-model knowledge is non-trivial. To this end, we subtly leverage the visual and linguistic-compatible feature character of ViT (i.e., student), and its capacity gap with the CNN (i.e., teacher) and propose a novel CNN-to-ViT KD framework, dubbed C2VKD. Importantly, as the teacher's features are heterogeneous to those of the student, we first propose a novel visual-linguistic feature distillation (VLFD) module that explores efficient KD among the aligned visual and linguistic-compatible representations. Moreover, due to the large capacity gap between the teacher and student and the inevitable prediction errors of the teacher, we then propose a pixel-wise decoupled distillation (PDD) module to supervise the student under the combination of labels and teacher's predictions from the decoupled target and non-target classes. Experiments on three semantic segmentation benchmark datasets consistently show that the increment of mIoU of our method is over 200% of the SoTA KD methods

{{</citation>}}


### (142/185) Uncovering Hidden Connections: Iterative Tracking and Reasoning for Video-grounded Dialog (Haoyu Zhang et al., 2023)

{{<citation>}}

Haoyu Zhang, Meng Liu, Yaowei Wang, Da Cao, Weili Guan, Liqiang Nie. (2023)  
**Uncovering Hidden Connections: Iterative Tracking and Reasoning for Video-grounded Dialog**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Dialog, GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.07259v1)  

---


**ABSTRACT**  
In contrast to conventional visual question answering, video-grounded dialog necessitates a profound understanding of both dialog history and video content for accurate response generation. Despite commendable strides made by existing methodologies, they often grapple with the challenges of incrementally understanding intricate dialog histories and assimilating video information. In response to this gap, we present an iterative tracking and reasoning strategy that amalgamates a textual encoder, a visual encoder, and a generator. At its core, our textual encoder is fortified with a path tracking and aggregation mechanism, adept at gleaning nuances from dialog history that are pivotal to deciphering the posed questions. Concurrently, our visual encoder harnesses an iterative reasoning network, meticulously crafted to distill and emphasize critical visual markers from videos, enhancing the depth of visual comprehension. Culminating this enriched information, we employ the pre-trained GPT-2 model as our response generator, stitching together coherent and contextually apt answers. Our empirical assessments, conducted on two renowned datasets, testify to the prowess and adaptability of our proposed design.

{{</citation>}}


### (143/185) ADASR: An Adversarial Auto-Augmentation Framework for Hyperspectral and Multispectral Data Fusion (Jinghui Qin et al., 2023)

{{<citation>}}

Jinghui Qin, Lihuang Fang, Ruitao Lu, Liang Lin, Yukai Shi. (2023)  
**ADASR: An Adversarial Auto-Augmentation Framework for Hyperspectral and Multispectral Data Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.07255v1)  

---


**ABSTRACT**  
Deep learning-based hyperspectral image (HSI) super-resolution, which aims to generate high spatial resolution HSI (HR-HSI) by fusing hyperspectral image (HSI) and multispectral image (MSI) with deep neural networks (DNNs), has attracted lots of attention. However, neural networks require large amounts of training data, hindering their application in real-world scenarios. In this letter, we propose a novel adversarial automatic data augmentation framework ADASR that automatically optimizes and augments HSI-MSI sample pairs to enrich data diversity for HSI-MSI fusion. Our framework is sample-aware and optimizes an augmentor network and two downsampling networks jointly by adversarial learning so that we can learn more robust downsampling networks for training the upsampling network. Extensive experiments on two public classical hyperspectral datasets demonstrate the effectiveness of our ADASR compared to the state-of-the-art methods.

{{</citation>}}


### (144/185) A Comparative Study of Pre-trained CNNs and GRU-Based Attention for Image Caption Generation (Rashid Khan et al., 2023)

{{<citation>}}

Rashid Khan, Bingding Huang, Haseeb Hassan, Asim Zaman, Zhongfu Ye. (2023)  
**A Comparative Study of Pre-trained CNNs and GRU-Based Attention for Image Caption Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.07252v1)  

---


**ABSTRACT**  
Image captioning is a challenging task involving generating a textual description for an image using computer vision and natural language processing techniques. This paper proposes a deep neural framework for image caption generation using a GRU-based attention mechanism. Our approach employs multiple pre-trained convolutional neural networks as the encoder to extract features from the image and a GRU-based language model as the decoder to generate descriptive sentences. To improve performance, we integrate the Bahdanau attention model with the GRU decoder to enable learning to focus on specific image parts. We evaluate our approach using the MSCOCO and Flickr30k datasets and show that it achieves competitive scores compared to state-of-the-art methods. Our proposed framework can bridge the gap between computer vision and natural language and can be extended to specific domains.

{{</citation>}}


### (145/185) Deep Learning for blind spectral unmixing of LULC classes with MODIS multispectral time series and ancillary data (José Rodríguez-Ortega et al., 2023)

{{<citation>}}

José Rodríguez-Ortega, Rohaifa Khaldi, Domingo Alcaraz-Segura, Siham Tabik. (2023)  
**Deep Learning for blind spectral unmixing of LULC classes with MODIS multispectral time series and ancillary data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.07223v1)  

---


**ABSTRACT**  
Remotely sensed data are dominated by mixed Land Use and Land Cover (LULC) types. Spectral unmixing is a technique to extract information from mixed pixels into their constituent LULC types and corresponding abundance fractions. Traditionally, solving this task has relied on either classical methods that require prior knowledge of endmembers or machine learning methods that avoid explicit endmembers calculation, also known as blind spectral unmixing (BSU). Most BSU studies based on Deep Learning (DL) focus on one time-step hyperspectral data, yet its acquisition remains quite costly compared with multispectral data. To our knowledge, here we provide the first study on BSU of LULC classes using multispectral time series data with DL models. We further boost the performance of a Long-Short Term Memory (LSTM)-based model by incorporating geographic plus topographic (geo-topographic) and climatic ancillary information. Our experiments show that combining spectral-temporal input data together with geo-topographic and climatic information substantially improves the abundance estimation of LULC classes in mixed pixels. To carry out this study, we built a new labeled dataset of the region of Andalusia (Spain) with monthly multispectral time series of pixels for the year 2013 from MODIS at 460m resolution, for two hierarchical levels of LULC classes, named Andalusia MultiSpectral MultiTemporal Unmixing (Andalusia-MSMTU). This dataset provides, at the pixel level, a multispectral time series plus ancillary information annotated with the abundance of each LULC class inside each pixel. The dataset and code are available to the public.

{{</citation>}}


### (146/185) Multiview Transformer: Rethinking Spatial Information in Hyperspectral Image Classification (Jie Zhang et al., 2023)

{{<citation>}}

Jie Zhang, Yongshan Zhang, Yicong Zhou. (2023)  
**Multiview Transformer: Rethinking Spatial Information in Hyperspectral Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07186v1)  

---


**ABSTRACT**  
Identifying the land cover category for each pixel in a hyperspectral image (HSI) relies on spectral and spatial information. An HSI cuboid with a specific patch size is utilized to extract spatial-spectral feature representation for the central pixel. In this article, we investigate that scene-specific but not essential correlations may be recorded in an HSI cuboid. This additional information improves the model performance on existing HSI datasets and makes it hard to properly evaluate the ability of a model. We refer to this problem as the spatial overfitting issue and utilize strict experimental settings to avoid it. We further propose a multiview transformer for HSI classification, which consists of multiview principal component analysis (MPCA), spectral encoder-decoder (SED), and spatial-pooling tokenization transformer (SPTT). MPCA performs dimension reduction on an HSI via constructing spectral multiview observations and applying PCA on each view data to extract low-dimensional view representation. The combination of view representations, named multiview representation, is the dimension reduction output of the MPCA. To aggregate the multiview information, a fully-convolutional SED with a U-shape in spectral dimension is introduced to extract a multiview feature map. SPTT transforms the multiview features into tokens using the spatial-pooling tokenization strategy and learns robust and discriminative spatial-spectral features for land cover identification. Classification is conducted with a linear classifier. Experiments on three HSI datasets with rigid settings demonstrate the superiority of the proposed multiview transformer over the state-of-the-art methods.

{{</citation>}}


### (147/185) Improving mitosis detection on histopathology images using large vision-language models (Ruiwen Ding et al., 2023)

{{<citation>}}

Ruiwen Ding, James Hall, Neil Tenenholtz, Kristen Severson. (2023)  
**Improving mitosis detection on histopathology images using large vision-language models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.07176v1)  

---


**ABSTRACT**  
In certain types of cancerous tissue, mitotic count has been shown to be associated with tumor proliferation, poor prognosis, and therapeutic resistance. Due to the high inter-rater variability of mitotic counting by pathologists, convolutional neural networks (CNNs) have been employed to reduce the subjectivity of mitosis detection in hematoxylin and eosin (H&E)-stained whole slide images. However, most existing models have performance that lags behind expert panel review and only incorporate visual information. In this work, we demonstrate that pre-trained large-scale vision-language models that leverage both visual features and natural language improve mitosis detection accuracy. We formulate the mitosis detection task as an image captioning task and a visual question answering (VQA) task by including metadata such as tumor and scanner types as context. The effectiveness of our pipeline is demonstrated via comparison with various baseline models using 9,501 mitotic figures and 11,051 hard negatives (non-mitotic figures that are difficult to characterize) from the publicly available Mitosis Domain Generalization Challenge (MIDOG22) dataset.

{{</citation>}}


## cs.HC (8)



### (148/185) Deepfakes, Phrenology, Surveillance, and More! A Taxonomy of AI Privacy Risks (Hao-Ping Lee et al., 2023)

{{<citation>}}

Hao-Ping Lee, Yu-Ju Yang, Thomas Serban von Davier, Jodi Forlizzi, Sauvik Das. (2023)  
**Deepfakes, Phrenology, Surveillance, and More! A Taxonomy of AI Privacy Risks**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07879v1)  

---


**ABSTRACT**  
Privacy is a key principle for developing ethical AI technologies, but how does including AI technologies in products and services change privacy risks? We constructed a taxonomy of AI privacy risks by analyzing 321 documented AI privacy incidents. We codified how the unique capabilities and requirements of AI technologies described in those incidents generated new privacy risks, exacerbated known ones, or otherwise did not meaningfully alter the risk. We present 12 high-level privacy risks that AI technologies either newly created (e.g., exposure risks from deepfake pornography) or exacerbated (e.g., surveillance risks from collecting training data). One upshot of our work is that incorporating AI technologies into a product can alter the privacy risks it entails. Yet, current privacy-preserving AI/ML methods (e.g., federated learning, differential privacy) only address a subset of the privacy risks arising from the capabilities and data requirements of AI.

{{</citation>}}


### (149/185) LLM4Vis: Explainable Visualization Recommendation using ChatGPT (Lei Wang et al., 2023)

{{<citation>}}

Lei Wang, Songheng Zhang, Yun Wang, Ee-Peng Lim, Yong Wang. (2023)  
**LLM4Vis: Explainable Visualization Recommendation using ChatGPT**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.07652v1)  

---


**ABSTRACT**  
Data visualization is a powerful tool for exploring and communicating insights in various domains. To automate visualization choice for datasets, a task known as visualization recommendation has been proposed. Various machine-learning-based approaches have been developed for this purpose, but they often require a large corpus of dataset-visualization pairs for training and lack natural explanations for their results. To address this research gap, we propose LLM4Vis, a novel ChatGPT-based prompting approach to perform visualization recommendation and return human-like explanations using very few demonstration examples. Our approach involves feature description, demonstration example selection, explanation generation, demonstration example construction, and inference steps. To obtain demonstration examples with high-quality explanations, we propose a new explanation generation bootstrapping to iteratively refine generated explanations by considering the previous generation and template-based hint. Evaluations on the VizML dataset show that LLM4Vis outperforms or performs similarly to supervised learning models like Random Forest, Decision Tree, and MLP in both few-shot and zero-shot settings. The qualitative evaluation also shows the effectiveness of explanations generated by LLM4Vis. We make our code publicly available at \href{https://github.com/demoleiwang/LLM4Vis}{https://github.com/demoleiwang/LLM4Vis}.

{{</citation>}}


### (150/185) Hypercomplex Multimodal Emotion Recognition from EEG and Peripheral Physiological Signals (Eleonora Lopez et al., 2023)

{{<citation>}}

Eleonora Lopez, Eleonora Chiarantano, Eleonora Grassucci, Danilo Comminiello. (2023)  
**Hypercomplex Multimodal Emotion Recognition from EEG and Peripheral Physiological Signals**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC, eess-SP  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.07648v1)  

---


**ABSTRACT**  
Multimodal emotion recognition from physiological signals is receiving an increasing amount of attention due to the impossibility to control them at will unlike behavioral reactions, thus providing more reliable information. Existing deep learning-based methods still rely on extracted handcrafted features, not taking full advantage of the learning ability of neural networks, and often adopt a single-modality approach, while human emotions are inherently expressed in a multimodal way. In this paper, we propose a hypercomplex multimodal network equipped with a novel fusion module comprising parameterized hypercomplex multiplications. Indeed, by operating in a hypercomplex domain the operations follow algebraic rules which allow to model latent relations among learned feature dimensions for a more effective fusion step. We perform classification of valence and arousal from electroencephalogram (EEG) and peripheral physiological signals, employing the publicly available database MAHNOB-HCI surpassing a multimodal state-of-the-art network. The code of our work is freely available at https://github.com/ispamm/MHyEEG.

{{</citation>}}


### (151/185) Qlarify: Bridging Scholarly Abstracts and Papers with Recursively Expandable Summaries (Raymond Fok et al., 2023)

{{<citation>}}

Raymond Fok, Joseph Chee Chang, Tal August, Amy X. Zhang, Daniel S. Weld. (2023)  
**Qlarify: Bridging Scholarly Abstracts and Papers with Recursively Expandable Summaries**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07581v1)  

---


**ABSTRACT**  
As scientific literature has grown exponentially, researchers often rely on paper triaging strategies such as browsing abstracts before deciding to delve into a paper's full text. However, when an abstract is insufficient, researchers are required to navigate an informational chasm between 150-word abstracts and 10,000-word papers. To bridge that gap, we introduce the idea of recursively expandable summaries and present Qlarify, an interactive system that allows users to recursively expand an abstract by progressively incorporating additional information from a paper's full text. Starting from an abstract, users can brush over summary text to specify targeted information needs or select AI-suggested entities in the text. Responses are then generated on-demand by an LLM and appear in the form of a fluid, threaded expansion of the existing text. Each generated summary can be efficiently verified through attribution to a relevant source-passage in the paper. Through an interview study (n=9) and a field deployment (n=275) at a research conference, we use Qlarify as a technology probe to elaborate upon the expandable summaries design space, highlight how scholars benefit from Qlarify's expandable abstracts, and identify future opportunities to support low-effort and just-in-time exploration of scientific documents $\unicode{x2013}$ and other information spaces $\unicode{x2013}$ through LLM-powered interactions.

{{</citation>}}


### (152/185) uxSense: Supporting User Experience Analysis with Visualization and Computer Vision (Andrea Batch et al., 2023)

{{<citation>}}

Andrea Batch, Yipeng Ji, Mingming Fan, Jian Zhao, Niklas Elmqvist. (2023)  
**uxSense: Supporting User Experience Analysis with Visualization and Computer Vision**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.07300v1)  

---


**ABSTRACT**  
Analyzing user behavior from usability evaluation can be a challenging and time-consuming task, especially as the number of participants and the scale and complexity of the evaluation grows. We propose uxSense, a visual analytics system using machine learning methods to extract user behavior from audio and video recordings as parallel time-stamped data streams. Our implementation draws on pattern recognition, computer vision, natural language processing, and machine learning to extract user sentiment, actions, posture, spoken words, and other features from such recordings. These streams are visualized as parallel timelines in a web-based front-end, enabling the researcher to search, filter, and annotate data across time and space. We present the results of a user study involving professional UX researchers evaluating user data using uxSense. In fact, we used uxSense itself to evaluate their sessions.

{{</citation>}}


### (153/185) Textiverse: A Scalable Visual Analytics System for Exploring Geotagged and Timestamped Text Corpora (Caroline Berger et al., 2023)

{{<citation>}}

Caroline Berger, Hanjun Xian, Krishna Madhavan, Niklas Elmqvist. (2023)  
**Textiverse: A Scalable Visual Analytics System for Exploring Geotagged and Timestamped Text Corpora**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Google, Twitter  
[Paper Link](http://arxiv.org/abs/2310.07242v1)  

---


**ABSTRACT**  
We propose Textiverse, a big data approach for mining geotagged timestamped textual data on a map, such as for Twitter feeds, crime reports, or restaurant reviews. We use a scalable data management pipeline that extracts keyphrases from online databases in parallel. We speed up this time-consuming step so that it outpaces the content creation rate of popular social media. The result is presented in a web-based interface that integrates with Google Maps to visualize textual content of massive scale. The visual design is based on aggregating spatial regions into discrete sites and rendering each such site as a circular tag cloud. To demonstrate the intended use of our technique, we first show how it can be used to characterize the U.S.\ National Science Foundation funding status based on all 489,151 awards. We then apply the same technique on visually representing a more spatially scattered and linguistically informal dataset: 1.2 million Twitter posts about the Android mobile operating system.

{{</citation>}}


### (154/185) 'Because Some Sighted People, They Don't Know What the Heck You're Talking About:' A Study of Blind TikTokers' Infrastructuring Work to Build Independence (Yao Lyu et al., 2023)

{{<citation>}}

Yao Lyu, John M. Carroll. (2023)  
**'Because Some Sighted People, They Don't Know What the Heck You're Talking About:' A Study of Blind TikTokers' Infrastructuring Work to Build Independence**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.07154v1)  

---


**ABSTRACT**  
There has been extensive research on the experiences of individuals with visual impairments on text- and image-based social media platforms, such as Facebook and Twitter. However, little is known about the experiences of visually impaired users on short-video platforms like TikTok. To bridge this gap, we conducted an interview study with 30 BlindTokers (the nickname of blind TikTokers). Our study aimed to explore the various activities of BlindTokers on TikTok, including everyday entertainment, professional development, and community engagement. The widespread usage of TikTok among participants demonstrated that they considered TikTok and its associated experiences as the infrastructure for their activities. Additionally, participants reported experiencing breakdowns in this infrastructure due to accessibility issues. They had to carry out infrastructuring work to resolve the breakdowns. Blind users' various practices on TikTok also foregrounded their perceptions of independence. We then discussed blind users' nuanced understanding of the TikTok-mediated independence; we also critically examined BlindTokers' infrastructuring work for such independence.

{{</citation>}}


### (155/185) An HCI-Centric Survey and Taxonomy of Human-Generative-AI Interactions (Jingyu Shi et al., 2023)

{{<citation>}}

Jingyu Shi, Rahul Jain, Hyungjun Doh, Ryo Suzuki, Karthik Ramani. (2023)  
**An HCI-Centric Survey and Taxonomy of Human-Generative-AI Interactions**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.07127v1)  

---


**ABSTRACT**  
Generative AI (GenAI) has shown remarkable capabilities in generating diverse and realistic content across different formats like images, videos, and text. In Generative AI, human involvement is essential, thus HCI literature has investigated how to effectively create collaborations between humans and GenAI systems. However, the current literature lacks a comprehensive framework to better understand Human-GenAI Interactions, as the holistic aspects of human-centered GenAI systems are rarely analyzed systematically. In this paper, we present a survey of 154 papers, providing a novel taxonomy and analysis of Human-GenAI Interactions from both human and Gen-AI perspectives. The dimension of design space includes 1) Purposes of Using Generative AI, 2) Feedback from Models to Users , 3) Control from Users to Models, 4) Levels of Engagement, 5) Application Domains, and 6) Evaluation Strategies. Our work is also timely at the current development stage of GenAI, where the Human-GenAI interaction design is of paramount importance. We also highlight challenges and opportunities to guide the design of Gen-AI systems and interactions towards the future design of human-centered Generative AI applications.

{{</citation>}}


## quant-ph (3)



### (156/185) QArchSearch: A Scalable Quantum Architecture Search Package (Ankit Kulshrestha et al., 2023)

{{<citation>}}

Ankit Kulshrestha, Danylo Lykov, Ilya Safro, Yuri Alexeev. (2023)  
**QArchSearch: A Scalable Quantum Architecture Search Package**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2310.07858v1)  

---


**ABSTRACT**  
The current era of quantum computing has yielded several algorithms that promise high computational efficiency. While the algorithms are sound in theory and can provide potentially exponential speedup, there is little guidance on how to design proper quantum circuits to realize the appropriate unitary transformation to be applied to the input quantum state. In this paper, we present \texttt{QArchSearch}, an AI based quantum architecture search package with the \texttt{QTensor} library as a backend that provides a principled and automated approach to finding the best model given a task and input quantum state. We show that the search package is able to efficiently scale the search to large quantum circuits and enables the exploration of more complex models for different quantum applications. \texttt{QArchSearch} runs at scale and high efficiency on high-performance computing systems using a two-level parallelization scheme on both CPUs and GPUs, which has been demonstrated on the Polaris supercomputer.

{{</citation>}}


### (157/185) Experimental quantum natural gradient optimization in photonics (Yizhi Wang et al., 2023)

{{<citation>}}

Yizhi Wang, Shichuan Xue, Yaxuan Wang, Jiangfang Ding, Weixu Shi, Dongyang Wang, Yong Liu, Yingwen Liu, Xiang Fu, Guangyao Huang, Anqi Huang, Mingtang Deng, Junjie Wu. (2023)  
**Experimental quantum natural gradient optimization in photonics**  

---
Primary Category: quant-ph  
Categories: cs-LG, physics-optics, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.07371v1)  

---


**ABSTRACT**  
Variational quantum algorithms (VQAs) combining the advantages of parameterized quantum circuits and classical optimizers, promise practical quantum applications in the Noisy Intermediate-Scale Quantum era. The performance of VQAs heavily depends on the optimization method. Compared with gradient-free and ordinary gradient descent methods, the quantum natural gradient (QNG), which mirrors the geometric structure of the parameter space, can achieve faster convergence and avoid local minima more easily, thereby reducing the cost of circuit executions. We utilized a fully programmable photonic chip to experimentally estimate the QNG in photonics for the first time. We obtained the dissociation curve of the He-H$^+$ cation and achieved chemical accuracy, verifying the outperformance of QNG optimization on a photonic device. Our work opens up a vista of utilizing QNG in photonics to implement practical near-term quantum applications.

{{</citation>}}


### (158/185) Unleashing quantum algorithms with Qinterpreter: bridging the gap between theory and practice across leading quantum computing platforms (Wilmer Contreras Sepúlveda et al., 2023)

{{<citation>}}

Wilmer Contreras Sepúlveda, Ángel David Torres-Palencia, José Javier Sánchez Mondragón, Braulio Misael Villegas-Martínez, J. Jesús Escobedo-Alatorre, Sandra Gesing, Néstor Lozano-Crisóstomo, Julio César García-Melgarejo, Juan Carlos Sánchez Pérez, Eddie Nelson Palacios- Pérez, Omar PalilleroSandoval. (2023)  
**Unleashing quantum algorithms with Qinterpreter: bridging the gap between theory and practice across leading quantum computing platforms**  

---
Primary Category: quant-ph  
Categories: cs-ET, quant-ph, quant-ph  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2310.07173v1)  

---


**ABSTRACT**  
Quantum computing is a rapidly emerging and promising field that has the potential to revolutionize numerous research domains, including drug design, network technologies and sustainable energy. Due to the inherent complexity and divergence from classical computing, several major quantum computing libraries have been developed to implement quantum algorithms, namely IBM Qiskit, Amazon Braket, Cirq, PyQuil, and PennyLane. These libraries allow for quantum simulations on classical computers and facilitate program execution on corresponding quantum hardware, e.g., Qiskit programs on IBM quantum computers. While all platforms have some differences, the main concepts are the same. QInterpreter is a tool embedded in the Quantum Science Gateway QubitHub using Jupyter Notebooks that translates seamlessly programs from one library to the other and visualizes the results. It combines the five well-known quantum libraries: into a unified framework. Designed as an educational tool for beginners, Qinterpreter enables the development and execution of quantum circuits across various platforms in a straightforward way. The work highlights the versatility and accessibility of Qinterpreter in quantum programming and underscores our ultimate goal of pervading Quantum Computing through younger, less specialized, and diverse cultural and national communities.

{{</citation>}}


## cs.AR (1)



### (159/185) DAG-aware Synthesis Orchestration (Yingjie Li et al., 2023)

{{<citation>}}

Yingjie Li, Mingju Liu, Mark Ren, Alan Mishchenko, Cunxi Yu. (2023)  
**DAG-aware Synthesis Orchestration**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07846v1)  

---


**ABSTRACT**  
The key methodologies of modern logic synthesis techniques are conducted on multi-level technology-independent representations such as And-Inverter-Graphs (AIGs) of the digital logic via directed-acyclic-graph (DAGs) traversal based structural rewriting, resubstitution, and refactoring. Existing state-of-the-art DAG-aware logic synthesis algorithms are all designed to perform stand-alone optimizations during a single DAG traversal. However, we empirically identify and demonstrate that these algorithms are limited in quality-of-results and runtime complexity due to this design concept. This work proposes Synthesis Orchestration, which orchestrates stand-alone operations within the single traversal of AIG. Thus, orchestration method explores more optimization opportunities and results in better performance. Our experimental results are comprehensively conducted on all 104 designs collected from ISCAS'85/89/99, VTR, and EPFL benchmark suites, with consistent logic minimization improvements over rewriting, resubstitution, refactoring, leading to an average of 4% more node reduction with improved runtime efficiency for the single optimization. Moreover, we evaluate orchestration as a plug-in algorithm in resyn and resyn3 flows in ABC, which demonstrates consistent logic minimization improvements (3.8% and 10.9% more node reduction on average). The runtime analysis demonstrates the orchestration outperforms stand-alone algorithms in both AIG minimization and runtime efficiency. Finally, we integrate the orchestration into OpenROAD for end-to-end performance evaluation. Our results demonstrate the advantages of the orchestration optimization technique, even after technology mapping and post-routing in the design flow have been conducted.

{{</citation>}}


## cs.IR (4)



### (160/185) Language Models As Semantic Indexers (Bowen Jin et al., 2023)

{{<citation>}}

Bowen Jin, Hansi Zeng, Guoyin Wang, Xiusi Chen, Tianxin Wei, Ruirui Li, Zhengyang Wang, Zheng Li, Yang Li, Hanqing Lu, Suhang Wang, Jiawei Han, Xianfeng Tang. (2023)  
**Language Models As Semantic Indexers**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07815v1)  

---


**ABSTRACT**  
Semantic identifier (ID) is an important concept in information retrieval that aims to preserve the semantics of objects such as documents and items inside their IDs. Previous studies typically adopt a two-stage pipeline to learn semantic IDs by first procuring embeddings using off-the-shelf text encoders and then deriving IDs based on the embeddings. However, each step introduces potential information loss and there is usually an inherent mismatch between the distribution of embeddings within the latent space produced by text encoders and the anticipated distribution required for semantic indexing. Nevertheless, it is non-trivial to design a method that can learn the document's semantic representations and its hierarchical structure simultaneously, given that semantic IDs are discrete and sequentially structured, and the semantic supervision is deficient. In this paper, we introduce LMINDEXER, a self-supervised framework to learn semantic IDs with a generative language model. We tackle the challenge of sequential discrete ID by introducing a semantic indexer capable of generating neural sequential discrete representations with progressive training and contrastive learning. In response to the semantic supervision deficiency, we propose to train the model with a self-supervised document reconstruction objective. The learned semantic indexer can facilitate various downstream tasks, such as recommendation and retrieval. We conduct experiments on three tasks including recommendation, product search, and document retrieval on five datasets from various domains, where LMINDEXER outperforms competitive baselines significantly and consistently.

{{</citation>}}


### (161/185) Retrieve Anything To Augment Large Language Models (Peitian Zhang et al., 2023)

{{<citation>}}

Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, Jian-Yun Nie. (2023)  
**Retrieve Anything To Augment Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07554v1)  

---


**ABSTRACT**  
Large language models (LLMs) face significant challenges stemming from the inherent limitations in knowledge, memory, alignment, and action. These challenges cannot be addressed by LLMs alone, but should rely on assistance from the external world, such as knowledge base, memory store, demonstration examples, and tools. Retrieval augmentation stands as a vital mechanism for bridging the gap between LLMs and the external assistance. However, conventional methods encounter two pressing issues. On one hand, the general-purpose retrievers are not properly optimized for the retrieval augmentation of LLMs. On the other hand, the task-specific retrievers lack the required versatility, hindering their performance across the diverse retrieval augmentation scenarios.   In this work, we present a novel approach, the LLM Embedder, which comprehensively support the diverse needs of LLMs' retrieval augmentation with one unified embedding model. Training such an unified model is non-trivial, as various retrieval tasks aim to capture distinct semantic relationships, often subject to mutual interference. To address this challenge, we systematically optimize our training methodology. This includes reward formulation based on LLMs' feedback, the stabilization of knowledge distillation, multi-task fine-tuning with explicit instructions, and the use of homogeneous in-batch negative sampling. These optimization strategies contribute to the outstanding empirical performance of the LLM-Embedder. Notably, it yields remarkable enhancements in retrieval augmentation for LLMs, surpassing both general-purpose and task-specific retrievers in various evaluation scenarios. This project is made publicly available at https://github.com/FlagOpen/FlagEmbedding.

{{</citation>}}


### (162/185) GMOCAT: A Graph-Enhanced Multi-Objective Method for Computerized Adaptive Testing (Hangyu Wang et al., 2023)

{{<citation>}}

Hangyu Wang, Ting Long, Liang Yin, Weinan Zhang, Wei Xia, Qichen Hong, Dingyin Xia, Ruiming Tang, Yong Yu. (2023)  
**GMOCAT: A Graph-Enhanced Multi-Objective Method for Computerized Adaptive Testing**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07477v1)  

---


**ABSTRACT**  
Computerized Adaptive Testing(CAT) refers to an online system that adaptively selects the best-suited question for students with various abilities based on their historical response records. Most CAT methods only focus on the quality objective of predicting the student ability accurately, but neglect concept diversity or question exposure control, which are important considerations in ensuring the performance and validity of CAT. Besides, the students' response records contain valuable relational information between questions and knowledge concepts. The previous methods ignore this relational information, resulting in the selection of sub-optimal test questions. To address these challenges, we propose a Graph-Enhanced Multi-Objective method for CAT (GMOCAT). Firstly, three objectives, namely quality, diversity and novelty, are introduced into the Scalarized Multi-Objective Reinforcement Learning framework of CAT, which respectively correspond to improving the prediction accuracy, increasing the concept diversity and reducing the question exposure. We use an Actor-Critic Recommender to select questions and optimize three objectives simultaneously by the scalarization function. Secondly, we utilize the graph neural network to learn relation-aware embeddings of questions and concepts. These embeddings are able to aggregate neighborhood information in the relation graphs between questions and concepts. We conduct experiments on three real-world educational datasets, and show that GMOCAT not only outperforms the state-of-the-art methods in the ability prediction, but also achieve superior performance in improving the concept diversity and alleviating the question exposure. Our code is available at https://github.com/justarter/GMOCAT.

{{</citation>}}


### (163/185) Preliminary Results of a Scientometric Analysis of the German Information Retrieval Community 2020-2023 (Philipp Schaer et al., 2023)

{{<citation>}}

Philipp Schaer, Svetlana Myshkina, Jüri Keller. (2023)  
**Preliminary Results of a Scientometric Analysis of the German Information Retrieval Community 2020-2023**  

---
Primary Category: cs.IR  
Categories: cs-DL, cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2310.07346v1)  

---


**ABSTRACT**  
The German Information Retrieval community is located in two different sub-fields: Information and computer science. There are no current studies that investigate these communities on a scientometric level. Available studies only focus on the information scientific part of the community. We generated a data set of 401 recent IR-related publications extracted from six core IR conferences from a mainly computer scientific background. We analyze this data set at the institutional and researcher level. The data set is publicly released, and we also demonstrate a mapping use case.

{{</citation>}}


## cs.CR (5)



### (164/185) DiPmark: A Stealthy, Efficient and Resilient Watermark for Large Language Models (Yihan Wu et al., 2023)

{{<citation>}}

Yihan Wu, Zhengmian Hu, Hongyang Zhang, Heng Huang. (2023)  
**DiPmark: A Stealthy, Efficient and Resilient Watermark for Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07710v1)  

---


**ABSTRACT**  
Watermarking techniques offer a promising way to secure data via embedding covert information into the data. A paramount challenge in the domain lies in preserving the distribution of original data during watermarking. Our research extends and refines existing watermarking framework, placing emphasis on the importance of a distribution-preserving (DiP) watermark. Contrary to the current strategies, our proposed DiPmark preserves the original token distribution during watermarking (stealthy), is detectable without access to the language model API or weights (efficient), and is robust to moderate changes of tokens (resilient). This is achieved by incorporating a novel reweight strategy, combined with a hash function that assigns unique \textit{i.i.d.} ciphers based on the context. The empirical benchmarks of our approach underscore its stealthiness, efficiency, and resilience, making it a robust solution for watermarking tasks that demand impeccable quality preservation.

{{</citation>}}


### (165/185) Composite Backdoor Attacks Against Large Language Models (Hai Huang et al., 2023)

{{<citation>}}

Hai Huang, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang. (2023)  
**Composite Backdoor Attacks Against Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: LLaMA, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.07676v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated superior performance compared to previous methods on various tasks, and often serve as the foundation models for many researches and services. However, the untrustworthy third-party LLMs may covertly introduce vulnerabilities for downstream tasks. In this paper, we explore the vulnerability of LLMs through the lens of backdoor attacks. Different from existing backdoor attacks against LLMs, ours scatters multiple trigger keys in different prompt components. Such a Composite Backdoor Attack (CBA) is shown to be stealthier than implanting the same multiple trigger keys in only a single component. CBA ensures that the backdoor is activated only when all trigger keys appear. Our experiments demonstrate that CBA is effective in both natural language processing (NLP) and multimodal tasks. For instance, with $3\%$ poisoning samples against the LLaMA-7B model on the Emotion dataset, our attack achieves a $100\%$ Attack Success Rate (ASR) with a False Triggered Rate (FTR) below $2.06\%$ and negligible model accuracy degradation. The unique characteristics of our CBA can be tailored for various practical scenarios, e.g., targeting specific user groups. Our work highlights the necessity of increased security research on the trustworthiness of foundation LLMs.

{{</citation>}}


### (166/185) My Brother Helps Me: Node Injection Based Adversarial Attack on Social Bot Detection (Lanjun Wang et al., 2023)

{{<citation>}}

Lanjun Wang, Xinran Qiao, Yanwei Xie, Weizhi Nie, Yongdong Zhang, Anan Liu. (2023)  
**My Brother Helps Me: Node Injection Based Adversarial Attack on Social Bot Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SI, cs.CR  
Keywords: Adversarial Attack, GNN, Twitter  
[Paper Link](http://arxiv.org/abs/2310.07159v1)  

---


**ABSTRACT**  
Social platforms such as Twitter are under siege from a multitude of fraudulent users. In response, social bot detection tasks have been developed to identify such fake users. Due to the structure of social networks, the majority of methods are based on the graph neural network(GNN), which is susceptible to attacks. In this study, we propose a node injection-based adversarial attack method designed to deceive bot detection models. Notably, neither the target bot nor the newly injected bot can be detected when a new bot is added around the target bot. This attack operates in a black-box fashion, implying that any information related to the victim model remains unknown. To our knowledge, this is the first study exploring the resilience of bot detection through graph node injection. Furthermore, we develop an attribute recovery module to revert the injected node embedding from the graph embedding space back to the original feature space, enabling the adversary to manipulate node perturbation effectively. We conduct adversarial attacks on four commonly used GNN structures for bot detection on two widely used datasets: Cresci-2015 and TwiBot-22. The attack success rate is over 73\% and the rate of newly injected nodes being detected as bots is below 13\% on these two datasets.

{{</citation>}}


### (167/185) No Privacy Left Outside: On the (In-)Security of TEE-Shielded DNN Partition for On-Device ML (Ziqi Zhang et al., 2023)

{{<citation>}}

Ziqi Zhang, Chen Gong, Yifeng Cai, Yuanyuan Yuan, Bingyan Liu, Ding Li, Yao Guo, Xiangqun Chen. (2023)  
**No Privacy Left Outside: On the (In-)Security of TEE-Shielded DNN Partition for On-Device ML**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.07152v1)  

---


**ABSTRACT**  
On-device ML introduces new security challenges: DNN models become white-box accessible to device users. Based on white-box information, adversaries can conduct effective model stealing (MS) and membership inference attack (MIA). Using Trusted Execution Environments (TEEs) to shield on-device DNN models aims to downgrade (easy) white-box attacks to (harder) black-box attacks. However, one major shortcoming is the sharply increased latency (up to 50X). To accelerate TEE-shield DNN computation with GPUs, researchers proposed several model partition techniques. These solutions, referred to as TEE-Shielded DNN Partition (TSDP), partition a DNN model into two parts, offloading the privacy-insensitive part to the GPU while shielding the privacy-sensitive part within the TEE. This paper benchmarks existing TSDP solutions using both MS and MIA across a variety of DNN models, datasets, and metrics. We show important findings that existing TSDP solutions are vulnerable to privacy-stealing attacks and are not as safe as commonly believed. We also unveil the inherent difficulty in deciding optimal DNN partition configurations (i.e., the highest security with minimal utility cost) for present TSDP solutions. The experiments show that such ``sweet spot'' configurations vary across datasets and models. Based on lessons harvested from the experiments, we present TEESlice, a novel TSDP method that defends against MS and MIA during DNN inference. TEESlice follows a partition-before-training strategy, which allows for accurate separation between privacy-related weights from public weights. TEESlice delivers the same security protection as shielding the entire DNN model inside TEE (the ``upper-bound'' security guarantees) with over 10X less overhead (in both experimental and real-world environments) than prior TSDP solutions and no accuracy loss.

{{</citation>}}


### (168/185) GraphCloak: Safeguarding Task-specific Knowledge within Graph-structured Data from Unauthorized Exploitation (Yixin Liu et al., 2023)

{{<citation>}}

Yixin Liu, Chenrui Fan, Xun Chen, Pan Zhou, Lichao Sun. (2023)  
**GraphCloak: Safeguarding Task-specific Knowledge within Graph-structured Data from Unauthorized Exploitation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.07100v1)  

---


**ABSTRACT**  
As Graph Neural Networks (GNNs) become increasingly prevalent in a variety of fields, from social network analysis to protein-protein interaction studies, growing concerns have emerged regarding the unauthorized utilization of personal data. Recent studies have shown that imperceptible poisoning attacks are an effective method of protecting image data from such misuse. However, the efficacy of this approach in the graph domain remains unexplored. To bridge this gap, this paper introduces GraphCloak to safeguard against the unauthorized usage of graph data. Compared with prior work, GraphCloak offers unique significant innovations: (1) graph-oriented, the perturbations are applied to both topological structures and descriptive features of the graph; (2) effective and stealthy, our cloaking method can bypass various inspections while causing a significant performance drop in GNNs trained on the cloaked graphs; and (3) stable across settings, our methods consistently perform effectively under a range of practical settings with limited knowledge. To address the intractable bi-level optimization problem, we propose two error-minimizing-based poisoning methods that target perturbations on the structural and feature space, along with a subgraph injection poisoning method. Our comprehensive evaluation of these methods underscores their effectiveness, stealthiness, and stability. We also delve into potential countermeasures and provide analytical justification for their effectiveness, paving the way for intriguing future research.

{{</citation>}}


## cs.CE (1)



### (169/185) Discovery of Novel Reticular Materials for Carbon Dioxide Capture using GFlowNets (Flaviu Cipcigan et al., 2023)

{{<citation>}}

Flaviu Cipcigan, Jonathan Booth, Rodrigo Neumann Barros Ferreira, Carine Ribeiro dos Santo, Mathias Steiner. (2023)  
**Discovery of Novel Reticular Materials for Carbon Dioxide Capture using GFlowNets**  

---
Primary Category: cs.CE  
Categories: cond-mat-mtrl-sci, cs-CE, cs.CE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07671v1)  

---


**ABSTRACT**  
Artificial intelligence holds promise to improve materials discovery. GFlowNets are an emerging deep learning algorithm with many applications in AI-assisted discovery. By using GFlowNets, we generate porous reticular materials, such as metal organic frameworks and covalent organic frameworks, for applications in carbon dioxide capture. We introduce a new Python package (matgfn) to train and sample GFlowNets. We use matgfn to generate the matgfn-rm dataset of novel and diverse reticular materials with gravimetric surface area above 5000 m$^2$/g. We calculate single- and two-component gas adsorption isotherms for the top-100 candidates in matgfn-rm. These candidates are novel compared to the state-of-art ARC-MOF dataset and rank in the 90th percentile in terms of working capacity compared to the CoRE2019 dataset. We discover 15 materials outperforming all materials in CoRE2019.

{{</citation>}}


## eess.IV (2)



### (170/185) Attention-Map Augmentation for Hypercomplex Breast Cancer Classification (Eleonora Lopez et al., 2023)

{{<citation>}}

Eleonora Lopez, Filippo Betello, Federico Carmignani, Eleonora Grassucci, Danilo Comminiello. (2023)  
**Attention-Map Augmentation for Hypercomplex Breast Cancer Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Augmentation  
[Paper Link](http://arxiv.org/abs/2310.07633v1)  

---


**ABSTRACT**  
Breast cancer is the most widespread neoplasm among women and early detection of this disease is critical. Deep learning techniques have become of great interest to improve diagnostic performance. Nonetheless, discriminating between malignant and benign masses from whole mammograms remains challenging due to them being almost identical to an untrained eye and the region of interest (ROI) occupying a minuscule portion of the entire image. In this paper, we propose a framework, parameterized hypercomplex attention maps (PHAM), to overcome these problems. Specifically, we deploy an augmentation step based on computing attention maps. Then, the attention maps are used to condition the classification step by constructing a multi-dimensional input comprised of the original breast cancer image and the corresponding attention map. In this step, a parameterized hypercomplex neural network (PHNN) is employed to perform breast cancer classification. The framework offers two main advantages. First, attention maps provide critical information regarding the ROI and allow the neural model to concentrate on it. Second, the hypercomplex architecture has the ability to model local relations between input dimensions thanks to hypercomplex algebra rules, thus properly exploiting the information provided by the attention map. We demonstrate the efficacy of the proposed framework on both mammography images as well as histopathological ones, surpassing attention-based state-of-the-art networks and the real-valued counterpart of our method. The code of our work is available at https://github.com/elelo22/AttentionBCS.

{{</citation>}}


### (171/185) PtychoDV: Vision Transformer-Based Deep Unrolling Network for Ptychographic Image Reconstruction (Weijie Gan et al., 2023)

{{<citation>}}

Weijie Gan, Qiuchen Zhai, Michael Thompson McCann, Cristina Garcia Cardona, Ulugbek S. Kamilov, Brendt Wohlberg. (2023)  
**PtychoDV: Vision Transformer-Based Deep Unrolling Network for Ptychographic Image Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.07504v1)  

---


**ABSTRACT**  
Ptychography is an imaging technique that captures multiple overlapping snapshots of a sample, illuminated coherently by a moving localized probe. The image recovery from ptychographic data is generally achieved via an iterative algorithm that solves a nonlinear phase-field problem derived from measured diffraction patterns. However, these approaches have high computational cost. In this paper, we introduce PtychoDV, a novel deep model-based network designed for efficient, high-quality ptychographic image reconstruction. PtychoDV comprises a vision transformer that generates an initial image from the set of raw measurements, taking into consideration their mutual correlations. This is followed by a deep unrolling network that refines the initial image using learnable convolutional priors and the ptychography measurement model. Experimental results on simulated data demonstrate that PtychoDV is capable of outperforming existing deep learning methods for this problem, and significantly reduces computational cost compared to iterative methodologies, while maintaining competitive performance.

{{</citation>}}


## cs.SI (2)



### (172/185) Analyzing Trendy Twitter Hashtags in the 2022 French Election (Aamir Mandviwalla et al., 2023)

{{<citation>}}

Aamir Mandviwalla, Lake Yin, Boleslaw K. Szymanski. (2023)  
**Analyzing Trendy Twitter Hashtags in the 2022 French Election**  

---
Primary Category: cs.SI  
Categories: cs-LG, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.07576v1)  

---


**ABSTRACT**  
Regressions trained to predict the future activity of social media users need rich features for accurate predictions. Many advanced models exist to generate such features; however, the time complexities of their computations are often prohibitive when they run on enormous data-sets. Some studies have shown that simple semantic network features can be rich enough to use for regressions without requiring complex computations. We propose a method for using semantic networks as user-level features for machine learning tasks. We conducted an experiment using a semantic network of 1037 Twitter hashtags from a corpus of 3.7 million tweets related to the 2022 French presidential election. A bipartite graph is formed where hashtags are nodes and weighted edges connect the hashtags reflecting the number of Twitter users that interacted with both hashtags. The graph is then transformed into a maximum-spanning tree with the most popular hashtag as its root node to construct a hierarchy amongst the hashtags. We then provide a vector feature for each user based on this tree. To validate the usefulness of our semantic feature we performed a regression experiment to predict the response rate of each user with six emotions like anger, enjoyment, or disgust. Our semantic feature performs well with the regression with most emotions having $R^2$ above 0.5. These results suggest that our semantic feature could be considered for use in further experiments predicting social media response on big data-sets.

{{</citation>}}


### (173/185) Generative Agent-Based Social Networks for Disinformation: Research Opportunities and Open Challenges (Javier Pastor-Galindo et al., 2023)

{{<citation>}}

Javier Pastor-Galindo, Pantaleone Nespoli, José A. Ruipérez-Valiente. (2023)  
**Generative Agent-Based Social Networks for Disinformation: Research Opportunities and Open Challenges**  

---
Primary Category: cs.SI  
Categories: cs-MA, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2310.07545v1)  

---


**ABSTRACT**  
This article presents the affordances that Generative Artificial Intelligence can have in disinformation context, one of the major threats to our digitalized society. We present a research framework to generate customized agent-based social networks for disinformation simulations that would enable understanding and evaluation of the phenomena whilst discussing open challenges.

{{</citation>}}


## math.AT (1)



### (174/185) ChatGPT for Computational Topology (Jian Liu et al., 2023)

{{<citation>}}

Jian Liu, Li Shen, Guo-Wei Wei. (2023)  
**ChatGPT for Computational Topology**  

---
Primary Category: math.AT  
Categories: cs-AI, math-AT, math.AT  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.07570v1)  

---


**ABSTRACT**  
ChatGPT represents a significant milestone in the field of artificial intelligence (AI), finding widespread applications across diverse domains. However, its effectiveness in mathematical contexts has been somewhat constrained by its susceptibility to conceptual errors. Concurrently, topological data analysis (TDA), a relatively new discipline, has garnered substantial interest in recent years. Nonetheless, the advancement of TDA is impeded by the limited understanding of computational algorithms and coding proficiency among theoreticians. This work endeavors to bridge the gap between theoretical topological concepts and their practical implementation in computational topology through the utilization of ChatGPT. We showcase how a pure theoretician, devoid of computational experience and coding skills, can effectively transform mathematical formulations and concepts into functional code for computational topology with the assistance of ChatGPT. Our strategy outlines a productive process wherein a mathematician trains ChatGPT on pure mathematical concepts, steers ChatGPT towards generating computational topology code, and subsequently validates the generated code using established examples. Our specific case studies encompass the computation of Betti numbers, Laplacian matrices, and Dirac matrices for simplicial complexes, as well as the persistence of various homologies and Laplacians. Furthermore, we explore the application of ChatGPT in computing recently developed topological theories for hypergraphs and digraphs. This work serves as an initial step towards effectively transforming pure mathematical theories into practical computational tools, with the ultimate goal of enabling real applications across diverse fields.

{{</citation>}}


## cs.CY (3)



### (175/185) Using Tableau and Google Map API for Understanding the Impact of Walkability on Dublin City (Minkun Kim, 2023)

{{<citation>}}

Minkun Kim. (2023)  
**Using Tableau and Google Map API for Understanding the Impact of Walkability on Dublin City**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY, stat-AP  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.07563v1)  

---


**ABSTRACT**  
In this article, we explore two effective means to communicate the concept of walkability - 1) visualization, and 2) descriptive statistics. We introduce the concept of walkability as measuring the quality of an urban space based on the distance needed to walk from that space to a range of different social, environmental, and economic amenities. We use Dublin city as a worked example and explore quantification and visualization of walkability of various areas of the city. We utilize the Google Map API and Tableau to visualize the less walkable areas across Dublin city and using WLS regression, we assess the effects of unwalkability on house prices in Dublin, thus quantifying the importance of walkable areas from an economic perspective.

{{</citation>}}


### (176/185) Energy Estimates Across Layers of Computing: From Devices to Large-Scale Applications in Machine Learning for Natural Language Processing, Scientific Computing, and Cryptocurrency Mining (Sadasivan Shankar, 2023)

{{<citation>}}

Sadasivan Shankar. (2023)  
**Energy Estimates Across Layers of Computing: From Devices to Large-Scale Applications in Machine Learning for Natural Language Processing, Scientific Computing, and Cryptocurrency Mining**  

---
Primary Category: cs.CY  
Categories: C-3; C-4; I-2; J-2, cs-AI, cs-CY, cs.CY  
Keywords: AI, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.07516v1)  

---


**ABSTRACT**  
Estimates of energy usage in layers of computing from devices to algorithms have been determined and analyzed. Building on the previous analysis [3], energy needed from single devices and systems including three large-scale computing applications such as Artificial Intelligence (AI)/Machine Learning for Natural Language Processing, Scientific Simulations, and Cryptocurrency Mining have been estimated. In contrast to the bit-level switching, in which transistors achieved energy efficiency due to geometrical scaling, higher energy is expended both at the at the instructions and simulations levels of an application. Additionally, the analysis based on AI/ML Accelerators indicate that changes in architectures using an older semiconductor technology node have comparable energy efficiency with a different architecture using a newer technology. Further comparisons of the energy in computing systems with the thermodynamic and biological limits, indicate that there is a 27-36 orders of magnitude higher energy requirements for total simulation of an application. These energy estimates underscore the need for serious considerations of energy efficiency in computing by including energy as a design parameter, enabling growing needs of compute-intensive applications in a digital world.

{{</citation>}}


### (177/185) ClausewitzGPT Framework: A New Frontier in Theoretical Large Language Model Enhanced Information Operations (Benjamin Kereopa-Yorke, 2023)

{{<citation>}}

Benjamin Kereopa-Yorke. (2023)  
**ClausewitzGPT Framework: A New Frontier in Theoretical Large Language Model Enhanced Information Operations**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CR, cs-CY, cs-SI, cs.CY  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07099v1)  

---


**ABSTRACT**  
In a digital epoch where cyberspace is the emerging nexus of geopolitical contention, the melding of information operations and Large Language Models (LLMs) heralds a paradigm shift, replete with immense opportunities and intricate challenges. As tools like the Mistral 7B LLM (Mistral, 2023) democratise access to LLM capabilities (Jin et al., 2023), a vast spectrum of actors, from sovereign nations to rogue entities (Howard et al., 2023), find themselves equipped with potent narrative-shaping instruments (Goldstein et al., 2023). This paper puts forth a framework for navigating this brave new world in the "ClausewitzGPT" equation. This novel formulation not only seeks to quantify the risks inherent in machine-speed LLM-augmented operations but also underscores the vital role of autonomous AI agents (Wang, Xie, et al., 2023). These agents, embodying ethical considerations (Hendrycks et al., 2021), emerge as indispensable components (Wang, Ma, et al., 2023), ensuring that as we race forward, we do not lose sight of moral compasses and societal imperatives.   Mathematically underpinned and inspired by the timeless tenets of Clausewitz's military strategy (Clausewitz, 1832), this thesis delves into the intricate dynamics of AI-augmented information operations. With references to recent findings and research (Department of State, 2023), it highlights the staggering year-on-year growth of AI information campaigns (Evgeny Pashentsev, 2023), stressing the urgency of our current juncture. The synthesis of Enlightenment thinking, and Clausewitz's principles provides a foundational lens, emphasising the imperative of clear strategic vision, ethical considerations, and holistic understanding in the face of rapid technological advancement.

{{</citation>}}


## eess.SP (1)



### (178/185) Uncovering ECG Changes during Healthy Aging using Explainable AI (Gabriel Ott et al., 2023)

{{<citation>}}

Gabriel Ott, Yannik Schaubelt, Juan Miguel Lopez Alcaraz, Wilhelm Haverkamp, Nils Strodthoff. (2023)  
**Uncovering ECG Changes during Healthy Aging using Explainable AI**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07463v1)  

---


**ABSTRACT**  
Cardiovascular diseases remain the leading global cause of mortality. This necessitates a profound understanding of heart aging processes to diagnose constraints in cardiovascular fitness. Traditionally, most of such insights have been drawn from the analysis of electrocardiogram (ECG) feature changes of individuals as they age. However, these features, while informative, may potentially obscure underlying data relationships. In this paper, we employ a deep-learning model and a tree-based model to analyze ECG data from a robust dataset of healthy individuals across varying ages in both raw signals and ECG feature format. Explainable AI techniques are then used to identify ECG features or raw signal characteristics are most discriminative for distinguishing between age groups. Our analysis with tree-based classifiers reveal age-related declines in inferred breathing rates and identifies notably high SDANN values as indicative of elderly individuals, distinguishing them from younger adults. Furthermore, the deep-learning model underscores the pivotal role of the P-wave in age predictions across all age groups, suggesting potential changes in the distribution of different P-wave types with age. These findings shed new light on age-related ECG changes, offering insights that transcend traditional feature-based approaches.

{{</citation>}}


## cs.IT (1)



### (179/185) WiGenAI: The Symphony of Wireless and Generative AI via Diffusion Models (Mehdi Letafati et al., 2023)

{{<citation>}}

Mehdi Letafati, Samad Ali, Matti Latva-aho. (2023)  
**WiGenAI: The Symphony of Wireless and Generative AI via Diffusion Models**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs-LG, cs.IT, math-IT  
Keywords: AI, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.07312v2)  

---


**ABSTRACT**  
Innovative foundation models, such as GPT-3 and stable diffusion models, have made a paradigm shift in the realm of artificial intelligence (AI) towards generative AI-based systems. In unison, from data communication and networking perspective, AI and machine learning (AI/ML) algorithms are envisioned to be pervasively incorporated into the future generations of wireless communications systems, highlighting the need for novel AI-native solutions for the emergent communication scenarios. In this article, we outline the applications of generative AI in wireless communication systems to lay the foundations for research in this field. Diffusion-based generative models, as the new state-of-the-art paradigm of generative models, are introduced, and their applications in wireless communication systems are discussed. Two case studies are also presented to showcase how diffusion models can be exploited for the development of resilient AI-native communication systems. Specifically, we propose denoising diffusion probabilistic models (DDPM) for a wireless communication scheme with non-ideal transceivers, where 30% improvement is achieved in terms of bit error rate. As the second application, DDPMs are employed at the transmitter to shape the constellation symbols, highlighting a robust out-of-distribution performance. Finally, future directions and open issues for the development of generative AI-based wireless systems are discussed to promote future research endeavors towards wireless generative AI (WiGenAI).

{{</citation>}}


## cs.MM (1)



### (180/185) Interactive Interior Design Recommendation via Coarse-to-fine Multimodal Reinforcement Learning (He Zhang et al., 2023)

{{<citation>}}

He Zhang, Ying Sun, Weiyu Guo, Yafei Liu, Haonan Lu, Xiaodong Lin, Hui Xiong. (2023)  
**Interactive Interior Design Recommendation via Coarse-to-fine Multimodal Reinforcement Learning**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07287v1)  

---


**ABSTRACT**  
Personalized interior decoration design often incurs high labor costs. Recent efforts in developing intelligent interior design systems have focused on generating textual requirement-based decoration designs while neglecting the problem of how to mine homeowner's hidden preferences and choose the proper initial design. To fill this gap, we propose an Interactive Interior Design Recommendation System (IIDRS) based on reinforcement learning (RL). IIDRS aims to find an ideal plan by interacting with the user, who provides feedback on the gap between the recommended plan and their ideal one. To improve decision-making efficiency and effectiveness in large decoration spaces, we propose a Decoration Recommendation Coarse-to-Fine Policy Network (DecorRCFN). Additionally, to enhance generalization in online scenarios, we propose an object-aware feedback generation method that augments model training with diversified and dynamic textual feedback. Extensive experiments on a real-world dataset demonstrate our method outperforms traditional methods by a large margin in terms of recommendation accuracy. Further user studies demonstrate that our method reaches higher real-world user satisfaction than baseline methods.

{{</citation>}}


## q-bio.QM (1)



### (181/185) Synthesizing Missing MRI Sequences from Available Modalities using Generative Adversarial Networks in BraTS Dataset (Ibrahim Ethem Hamamci, 2023)

{{<citation>}}

Ibrahim Ethem Hamamci. (2023)  
**Synthesizing Missing MRI Sequences from Available Modalities using Generative Adversarial Networks in BraTS Dataset**  

---
Primary Category: q-bio.QM  
Categories: cs-CV, cs-LG, eess-IV, q-bio-QM, q-bio.QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07250v1)  

---


**ABSTRACT**  
Glioblastoma is a highly aggressive and lethal form of brain cancer. Magnetic resonance imaging (MRI) plays a significant role in the diagnosis, treatment planning, and follow-up of glioblastoma patients due to its non-invasive and radiation-free nature. The International Brain Tumor Segmentation (BraTS) challenge has contributed to generating numerous AI algorithms to accurately and efficiently segment glioblastoma sub-compartments using four structural (T1, T1Gd, T2, T2-FLAIR) MRI scans. However, these four MRI sequences may not always be available. To address this issue, Generative Adversarial Networks (GANs) can be used to synthesize the missing MRI sequences. In this paper, we implement and utilize an open-source GAN approach that takes any three MRI sequences as input to generate the missing fourth structural sequence. Our proposed approach is contributed to the community-driven generally nuanced deep learning framework (GaNDLF) and demonstrates promising results in synthesizing high-quality and realistic MRI sequences, enabling clinicians to improve their diagnostic capabilities and support the application of AI methods to brain tumor MRI quantification.

{{</citation>}}


## cs.MA (1)



### (182/185) Quantifying Agent Interaction in Multi-agent Reinforcement Learning for Cost-efficient Generalization (Yuxin Chen et al., 2023)

{{<citation>}}

Yuxin Chen, Chen Tang, Ran Tian, Chenran Li, Jinning Li, Masayoshi Tomizuka, Wei Zhan. (2023)  
**Quantifying Agent Interaction in Multi-agent Reinforcement Learning for Cost-efficient Generalization**  

---
Primary Category: cs.MA  
Categories: I-2-6, cs-AI, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07218v1)  

---


**ABSTRACT**  
Generalization poses a significant challenge in Multi-agent Reinforcement Learning (MARL). The extent to which an agent is influenced by unseen co-players depends on the agent's policy and the specific scenario. A quantitative examination of this relationship sheds light on effectively training agents for diverse scenarios. In this study, we present the Level of Influence (LoI), a metric quantifying the interaction intensity among agents within a given scenario and environment. We observe that, generally, a more diverse set of co-play agents during training enhances the generalization performance of the ego agent; however, this improvement varies across distinct scenarios and environments. LoI proves effective in predicting these improvement disparities within specific scenarios. Furthermore, we introduce a LoI-guided resource allocation method tailored to train a set of policies for diverse scenarios under a constrained budget. Our results demonstrate that strategic resource allocation based on LoI can achieve higher performance than uniform allocation under the same computation budget.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (183/185) MatChat: A Large Language Model and Application Service Platform for Materials Science (Ziyi Chen et al., 2023)

{{<citation>}}

Ziyi Chen, Fankai Xie, Meng Wan, Yang Yuan, Miao Liu, Zongguo Wang, Sheng Meng, Yangang Wang. (2023)  
**MatChat: A Large Language Model and Application Service Platform for Materials Science**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-AI  
Keywords: AI, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07197v1)  

---


**ABSTRACT**  
The prediction of chemical synthesis pathways plays a pivotal role in materials science research. Challenges, such as the complexity of synthesis pathways and the lack of comprehensive datasets, currently hinder our ability to predict these chemical processes accurately. However, recent advancements in generative artificial intelligence (GAI), including automated text generation and question-answering systems, coupled with fine-tuning techniques, have facilitated the deployment of large-scale AI models tailored to specific domains. In this study, we harness the power of the LLaMA2-7B model and enhance it through a learning process that incorporates 13,878 pieces of structured material knowledge data. This specialized AI model, named MatChat, focuses on predicting inorganic material synthesis pathways. MatChat exhibits remarkable proficiency in generating and reasoning with knowledge in materials science. Although MatChat requires further refinement to meet the diverse material design needs, this research undeniably highlights its impressive reasoning capabilities and innovative potential in the field of materials science. MatChat is now accessible online and open for use, with both the model and its application framework available as open source. This study establishes a robust foundation for collaborative innovation in the integration of generative AI in materials science.

{{</citation>}}


## cs.SD (2)



### (184/185) Psychoacoustic Challenges Of Speech Enhancement On VoIP Platforms (Joseph Konan et al., 2023)

{{<citation>}}

Joseph Konan, Ojas Bhargave, Shikhar Agnihotri, Shuo Han, Yunyang Zeng, Ankit Shah, Bhiksha Raj. (2023)  
**Psychoacoustic Challenges Of Speech Enhancement On VoIP Platforms**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.07161v1)  

---


**ABSTRACT**  
Within the ambit of VoIP (Voice over Internet Protocol) telecommunications, the complexities introduced by acoustic transformations merit rigorous analysis. This research, rooted in the exploration of proprietary sender-side denoising effects, meticulously evaluates platforms such as Google Meets and Zoom. The study draws upon the Deep Noise Suppression (DNS) 2020 dataset, ensuring a structured examination tailored to various denoising settings and receiver interfaces. A methodological novelty is introduced via the Oaxaca decomposition, traditionally an econometric tool, repurposed herein to analyze acoustic-phonetic perturbations within VoIP systems. To further ground the implications of these transformations, psychoacoustic metrics, specifically PESQ and STOI, were harnessed to furnish a comprehensive understanding of speech alterations. Cumulatively, the insights garnered underscore the intricate landscape of VoIP-influenced acoustic dynamics. In addition to the primary findings, a multitude of metrics are reported, extending the research purview. Moreover, out-of-domain benchmarking for both time and time-frequency domain speech enhancement models is included, thereby enhancing the depth and applicability of this inquiry.

{{</citation>}}


### (185/185) LLark: A Multimodal Foundation Model for Music (Josh Gardner et al., 2023)

{{<citation>}}

Josh Gardner, Simon Durand, Daniel Stoller, Rachel M. Bittner. (2023)  
**LLark: A Multimodal Foundation Model for Music**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07160v1)  

---


**ABSTRACT**  
Music has a unique and complex structure which is challenging for both expert humans and existing AI systems to understand, and presents unique challenges relative to other forms of audio. We present LLark, an instruction-tuned multimodal model for music understanding. We detail our process for dataset creation, which involves augmenting the annotations of diverse open-source music datasets and converting them to a unified instruction-tuning format. We propose a multimodal architecture for LLark, integrating a pretrained generative model for music with a pretrained language model. In evaluations on three types of tasks (music understanding, captioning, and reasoning), we show that our model matches or outperforms existing baselines in zero-shot generalization for music understanding, and that humans show a high degree of agreement with the model's responses in captioning and reasoning tasks. LLark is trained entirely from open-source music data and models, and we make our training code available along with the release of this paper. Additional results and audio examples are at https://bit.ly/llark, and our source code is available at https://github.com/spotify-research/llark .

{{</citation>}}
