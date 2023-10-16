---
draft: false
title: "arXiv @ 2023.10.14"
date: 2023-10-14
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.14"
    identifier: arxiv_20231014
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (8)](#csro-8)
- [cs.AI (14)](#csai-14)
- [cs.CV (31)](#cscv-31)
- [cond-mat.mes-hall (1)](#cond-matmes-hall-1)
- [cs.CL (47)](#cscl-47)
- [cs.LG (36)](#cslg-36)
- [cs.SD (2)](#cssd-2)
- [cs.SI (2)](#cssi-2)
- [cs.SE (3)](#csse-3)
- [cs.HC (3)](#cshc-3)
- [econ.EM (1)](#econem-1)
- [cs.GR (1)](#csgr-1)
- [cs.NI (2)](#csni-2)
- [cs.CY (1)](#cscy-1)
- [physics.comp-ph (1)](#physicscomp-ph-1)
- [cs.DC (4)](#csdc-4)
- [eess.SY (2)](#eesssy-2)
- [stat.ML (2)](#statml-2)
- [eess.AS (2)](#eessas-2)
- [cs.CR (2)](#cscr-2)
- [cs.IR (1)](#csir-1)
- [eess.SP (2)](#eesssp-2)
- [eess.IV (1)](#eessiv-1)
- [cs.DS (1)](#csds-1)
- [cs.GT (1)](#csgt-1)
- [q-bio.BM (1)](#q-biobm-1)

## cs.RO (8)



### (1/172) 3D Self-Localization of Drones using a Single Millimeter-Wave Anchor (Maisy Lam et al., 2023)

{{<citation>}}

Maisy Lam, Laura Dodds, Aline Eid, Jimmy Hester, Fadel Adib. (2023)  
**3D Self-Localization of Drones using a Single Millimeter-Wave Anchor**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.08778v1)  

---


**ABSTRACT**  
We present the design, implementation, and evaluation of MiFly, a self-localization system for autonomous drones that works across indoor and outdoor environments, including low-visibility, dark, and GPS-denied settings. MiFly performs 6DoF self-localization by leveraging a single millimeter-wave (mmWave) anchor in its vicinity - even if that anchor is visually occluded. MmWave signals are used in radar and 5G systems and can operate in the dark and through occlusions. MiFly introduces a new mmWave anchor design and mounts light-weight high-resolution mmWave radars on a drone. By jointly designing the localization algorithms and the novel low-power mmWave anchor hardware (including its polarization and modulation), the drone is capable of high-speed 3D localization. Furthermore, by intelligently fusing the location estimates from its mmWave radars and its IMUs, it can accurately and robustly track its 6DoF trajectory. We implemented and evaluated MiFly on a DJI drone. We demonstrate a median localization error of 7cm and a 90th percentile less than 15cm, even when the anchor is fully occluded (visually) from the drone.

{{</citation>}}


### (2/172) Multi-Robot IMU Preintegration in the Presence of Bias and Communication Constraints (Mohammed Ayman Shalaby et al., 2023)

{{<citation>}}

Mohammed Ayman Shalaby, Charles Champagne Cossette, Jerome Le Ny, James Richard Forbes. (2023)  
**Multi-Robot IMU Preintegration in the Presence of Bias and Communication Constraints**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.08686v1)  

---


**ABSTRACT**  
This document is in supplement to the paper titled "Multi-Robot Relative Pose Estimation and IMU Preintegration Using Passive UWB Transceivers", available at [1]. The purpose of this document is to show how IMU biases can be incorporated into the framework presented in [1], while maintaining the differential Sylvester equation form of the process model.

{{</citation>}}


### (3/172) Pay Attention to How You Drive: Safe and Adaptive Model-Based Reinforcement Learning for Off-Road Driving (Sean J. Wang et al., 2023)

{{<citation>}}

Sean J. Wang, Honghao Zhu, Aaron M. Johnson. (2023)  
**Pay Attention to How You Drive: Safe and Adaptive Model-Based Reinforcement Learning for Off-Road Driving**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention, Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08674v1)  

---


**ABSTRACT**  
Autonomous off-road driving is challenging as risky actions taken by the robot may lead to catastrophic damage. As such, developing controllers in simulation is often desirable as it provides a safer and more economical alternative. However, accurately modeling robot dynamics is difficult due to the complex robot dynamics and terrain interactions in unstructured environments. Domain randomization addresses this problem by randomizing simulation dynamics parameters, however this approach sacrifices performance for robustness leading to policies that are sub-optimal for any target dynamics. We introduce a novel model-based reinforcement learning approach that aims to balance robustness with adaptability. Our approach trains a System Identification Transformer (SIT) and an Adaptive Dynamics Model (ADM) under a variety of simulated dynamics. The SIT uses attention mechanisms to distill state-transition observations from the target system into a context vector, which provides an abstraction for its target dynamics. Conditioned on this, the ADM probabilistically models the system's dynamics. Online, we use a Risk-Aware Model Predictive Path Integral controller (MPPI) to safely control the robot under its current understanding of the dynamics. We demonstrate in simulation as well as in multiple real-world environments that this approach enables safer behaviors upon initialization and becomes less conservative (i.e. faster) as its understanding of the target system dynamics improves with more observations. In particular, our approach results in an approximately 41% improvement in lap-time over the non-adaptive baseline while remaining safe across different environments.

{{</citation>}}


### (4/172) PolyTask: Learning Unified Policies through Behavior Distillation (Siddhant Haldar et al., 2023)

{{<citation>}}

Siddhant Haldar, Lerrel Pinto. (2023)  
**PolyTask: Learning Unified Policies through Behavior Distillation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.08573v1)  

---


**ABSTRACT**  
Unified models capable of solving a wide variety of tasks have gained traction in vision and NLP due to their ability to share regularities and structures across tasks, which improves individual task performance and reduces computational footprint. However, the impact of such models remains limited in embodied learning problems, which present unique challenges due to interactivity, sample inefficiency, and sequential task presentation. In this work, we present PolyTask, a novel method for learning a single unified model that can solve various embodied tasks through a 'learn then distill' mechanism. In the 'learn' step, PolyTask leverages a few demonstrations for each task to train task-specific policies. Then, in the 'distill' step, task-specific policies are distilled into a single policy using a new distillation method called Behavior Distillation. Given a unified policy, individual task behavior can be extracted through conditioning variables. PolyTask is designed to be conceptually simple while being able to leverage well-established algorithms in RL to enable interactivity, a handful of expert demonstrations to allow for sample efficiency, and preventing interactive access to tasks during distillation to enable lifelong learning. Experiments across three simulated environment suites and a real-robot suite show that PolyTask outperforms prior state-of-the-art approaches in multi-task and lifelong learning settings by significant margins.

{{</citation>}}


### (5/172) Security Considerations in AI-Robotics: A Survey of Current Methods, Challenges, and Opportunities (Subash Neupane et al., 2023)

{{<citation>}}

Subash Neupane, Shaswata Mitra, Ivan A. Fernandez, Swayamjit Saha, Sudip Mittal, Jingdao Chen, Nisha Pillai, Shahram Rahimi. (2023)  
**Security Considerations in AI-Robotics: A Survey of Current Methods, Challenges, and Opportunities**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2310.08565v1)  

---


**ABSTRACT**  
Robotics and Artificial Intelligence (AI) have been inextricably intertwined since their inception. Today, AI-Robotics systems have become an integral part of our daily lives, from robotic vacuum cleaners to semi-autonomous cars. These systems are built upon three fundamental architectural elements: perception, navigation and planning, and control. However, while the integration of AI-Robotics systems has enhanced the quality our lives, it has also presented a serious problem - these systems are vulnerable to security attacks. The physical components, algorithms, and data that make up AI-Robotics systems can be exploited by malicious actors, potentially leading to dire consequences. Motivated by the need to address the security concerns in AI-Robotics systems, this paper presents a comprehensive survey and taxonomy across three dimensions: attack surfaces, ethical and legal concerns, and Human-Robot Interaction (HRI) security. Our goal is to provide users, developers and other stakeholders with a holistic understanding of these areas to enhance the overall AI-Robotics system security. We begin by surveying potential attack surfaces and provide mitigating defensive strategies. We then delve into ethical issues, such as dependency and psychological impact, as well as the legal concerns regarding accountability for these systems. Besides, emerging trends such as HRI are discussed, considering privacy, integrity, safety, trustworthiness, and explainability concerns. Finally, we present our vision for future research directions in this dynamic and promising field.

{{</citation>}}


### (6/172) ALPHA: Attention-based Long-horizon Pathfinding in Highly-structured Areas (Chengyang He et al., 2023)

{{<citation>}}

Chengyang He, Tianze Yang, Tanishq Duhan, Yutong Wang, Guillaume Sartoretti. (2023)  
**ALPHA: Attention-based Long-horizon Pathfinding in Highly-structured Areas**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08350v1)  

---


**ABSTRACT**  
The multi-agent pathfinding (MAPF) problem seeks collision-free paths for a team of agents from their current positions to their pre-set goals in a known environment, and is an essential problem found at the core of many logistics, transportation, and general robotics applications. Existing learning-based MAPF approaches typically only let each agent make decisions based on a limited field-of-view (FOV) around its position, as a natural means to fix the input dimensions of its policy network. However, this often makes policies short-sighted, since agents lack the ability to perceive and plan for obstacles/agents beyond their FOV. To address this challenge, we propose ALPHA, a new framework combining the use of ground truth proximal (local) information and fuzzy distal (global) information to let agents sequence local decisions based on the full current state of the system, and avoid such myopicity. We further allow agents to make short-term predictions about each others' paths, as a means to reason about each others' path intentions, thereby enhancing the level of cooperation among agents at the whole system level. Our neural structure relies on a Graph Transformer architecture to allow agents to selectively combine these different sources of information and reason about their inter-dependencies at different spatial scales. Our simulation experiments demonstrate that ALPHA outperforms both globally-guided MAPF solvers and communication-learning based ones, showcasing its potential towards scalability in realistic deployments.

{{</citation>}}


### (7/172) Hilbert Space Embedding-based Trajectory Optimization for Multi-Modal Uncertain Obstacle Trajectory Prediction (Basant Sharma et al., 2023)

{{<citation>}}

Basant Sharma, Aditya Sharma, K. Madhava Krishna, Arun Kumar Singh. (2023)  
**Hilbert Space Embedding-based Trajectory Optimization for Multi-Modal Uncertain Obstacle Trajectory Prediction**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.08270v1)  

---


**ABSTRACT**  
Safe autonomous driving critically depends on how well the ego-vehicle can predict the trajectories of neighboring vehicles. To this end, several trajectory prediction algorithms have been presented in the existing literature. Many of these approaches output a multi-modal distribution of obstacle trajectories instead of a single deterministic prediction to account for the underlying uncertainty. However, existing planners cannot handle the multi-modality based on just sample-level information of the predictions. With this motivation, this paper proposes a trajectory optimizer that can leverage the distributional aspects of the prediction in a computationally tractable and sample-efficient manner. Our optimizer can work with arbitrarily complex distributions and thus can be used with output distribution represented as a deep neural network. The core of our approach is built on embedding distribution in Reproducing Kernel Hilbert Space (RKHS), which we leverage in two ways. First, we propose an RKHS embedding approach to select probable samples from the obstacle trajectory distribution. Second, we rephrase chance-constrained optimization as distribution matching in RKHS and propose a novel sampling-based optimizer for its solution. We validate our approach with hand-crafted and neural network-based predictors trained on real-world datasets and show improvement over the existing stochastic optimization approaches in safety metrics.

{{</citation>}}


### (8/172) Think, Act, and Ask: Open-World Interactive Personalized Robot Navigation (Yinpei Dai et al., 2023)

{{<citation>}}

Yinpei Dai, Run Peng, Sikai Li, Joyce Chai. (2023)  
**Think, Act, and Ask: Open-World Interactive Personalized Robot Navigation**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-HC, cs-RO, cs.RO  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.07968v1)  

---


**ABSTRACT**  
Zero-Shot Object Navigation (ZSON) enables agents to navigate towards open-vocabulary objects in unknown environments. The existing works of ZSON mainly focus on following individual instructions to find generic object classes, neglecting the utilization of natural language interaction and the complexities of identifying user-specific objects. To address these limitations, we introduce Zero-shot Interactive Personalized Object Navigation (ZIPON), where robots need to navigate to personalized goal objects while engaging in conversations with users. To solve ZIPON, we propose a new framework termed Open-woRld Interactive persOnalized Navigation (ORION), which uses Large Language Models (LLMs) to make sequential decisions to manipulate different modules for perception, navigation and communication. Experimental results show that the performance of interactive agents that can leverage user feedback exhibits significant improvement. However, obtaining a good balance between task completion and the efficiency of navigation and interaction remains challenging for all methods. We further provide more findings on the impact of diverse user feedback forms on the agents' performance.

{{</citation>}}


## cs.AI (14)



### (9/172) Examining the Potential and Pitfalls of ChatGPT in Science and Engineering Problem-Solving (Karen D. Wang et al., 2023)

{{<citation>}}

Karen D. Wang, Eric Burkholder, Carl Wieman, Shima Salehi, Nick Haber. (2023)  
**Examining the Potential and Pitfalls of ChatGPT in Science and Engineering Problem-Solving**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs.AI  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.08773v1)  

---


**ABSTRACT**  
The study explores the capabilities of OpenAI's ChatGPT in solving different types of physics problems. ChatGPT (with GPT-4) was queried to solve a total of 40 problems from a college-level engineering physics course. These problems ranged from well-specified problems, where all data required for solving the problem was provided, to under-specified, real-world problems where not all necessary data were given. Our findings show that ChatGPT could successfully solve 62.5\% of the well-specified problems, but its accuracy drops to 8.3\% for under-specified problems. Analysis of the model's incorrect solutions revealed three distinct failure modes: 1) failure to construct accurate models of the physical world, 2) failure to make reasonable assumptions about missing data, and 3) calculation errors. The study offers implications for how to leverage LLM-augmented instructional materials to enhance STEM education. The insights also contribute to the broader discourse on AI's strengths and limitations, serving both educators aiming to leverage the technology and researchers investigating human-AI collaboration frameworks for problem-solving and decision-making.

{{</citation>}}


### (10/172) Real-Time Event Detection with Random Forests and Temporal Convolutional Networks for More Sustainable Petroleum Industry (Yuanwei Qu et al., 2023)

{{<citation>}}

Yuanwei Qu, Baifan Zhou, Arild Waaler, David Cameron. (2023)  
**Real-Time Event Detection with Random Forests and Temporal Convolutional Networks for More Sustainable Petroleum Industry**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Event Detection  
[Paper Link](http://arxiv.org/abs/2310.08737v1)  

---


**ABSTRACT**  
The petroleum industry is crucial for modern society, but the production process is complex and risky. During the production, accidents or failures, resulting from undesired production events, can cause severe environmental and economic damage. Previous studies have investigated machine learning (ML) methods for undesired event detection. However, the prediction of event probability in real-time was insufficiently addressed, which is essential since it is important to undertake early intervention when an event is expected to happen. This paper proposes two ML approaches, random forests and temporal convolutional networks, to detect undesired events in real-time. Results show that our approaches can effectively classify event types and predict the probability of their appearance, addressing the challenges uncovered in previous studies and providing a more effective solution for failure event management during the production.

{{</citation>}}


### (11/172) A Lightweight Calibrated Simulation Enabling Efficient Offline Learning for Optimal Control of Real Buildings (Judah Goldfeder et al., 2023)

{{<citation>}}

Judah Goldfeder, John Sipple. (2023)  
**A Lightweight Calibrated Simulation Enabling Efficient Offline Learning for Optimal Control of Real Buildings**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs.AI, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08569v1)  

---


**ABSTRACT**  
Modern commercial Heating, Ventilation, and Air Conditioning (HVAC) devices form a complex and interconnected thermodynamic system with the building and outside weather conditions, and current setpoint control policies are not fully optimized for minimizing energy use and carbon emission. Given a suitable training environment, a Reinforcement Learning (RL) model is able to improve upon these policies, but training such a model, especially in a way that scales to thousands of buildings, presents many real world challenges. We propose a novel simulation-based approach, where a customized simulator is used to train the agent for each building. Our open-source simulator (available online: https://github.com/google/sbsim) is lightweight and calibrated via telemetry from the building to reach a higher level of fidelity. On a two-story, 68,000 square foot building, with 127 devices, we were able to calibrate our simulator to have just over half a degree of drift from the real world over a six-hour interval. This approach is an important step toward having a real-world RL control system that can be scaled to many buildings, allowing for greater efficiency and resulting in reduced energy consumption and carbon emissions.

{{</citation>}}


### (12/172) MemGPT: Towards LLMs as Operating Systems (Charles Packer et al., 2023)

{{<citation>}}

Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, Joseph E. Gonzalez. (2023)  
**MemGPT: Towards LLMs as Operating Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2310.08560v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized AI, but are constrained by limited context windows, hindering their utility in tasks like extended conversations and document analysis. To enable using context beyond limited context windows, we propose virtual context management, a technique drawing inspiration from hierarchical memory systems in traditional operating systems that provide the appearance of large memory resources through data movement between fast and slow memory. Using this technique, we introduce MemGPT (Memory-GPT), a system that intelligently manages different memory tiers in order to effectively provide extended context within the LLM's limited context window, and utilizes interrupts to manage control flow between itself and the user. We evaluate our OS-inspired design in two domains where the limited context windows of modern LLMs severely handicaps their performance: document analysis, where MemGPT is able to analyze large documents that far exceed the underlying LLM's context window, and multi-session chat, where MemGPT can create conversational agents that remember, reflect, and evolve dynamically through long-term interactions with their users. We release MemGPT code and data for our experiments at https://memgpt.ai.

{{</citation>}}


### (13/172) The Impact of Explanations on Fairness in Human-AI Decision-Making: Protected vs Proxy Features (Navita Goyal et al., 2023)

{{<citation>}}

Navita Goyal, Connor Baumler, Tin Nguyen, Hal Daumé III. (2023)  
**The Impact of Explanations on Fairness in Human-AI Decision-Making: Protected vs Proxy Features**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08617v1)  

---


**ABSTRACT**  
AI systems have been known to amplify biases in real world data. Explanations may help human-AI teams address these biases for fairer decision-making. Typically, explanations focus on salient input features. If a model is biased against some protected group, explanations may include features that demonstrate this bias, but when biases are realized through proxy features, the relationship between this proxy feature and the protected one may be less clear to a human. In this work, we study the effect of the presence of protected and proxy features on participants' perception of model fairness and their ability to improve demographic parity over an AI alone. Further, we examine how different treatments -- explanations, model bias disclosure and proxy correlation disclosure -- affect fairness perception and parity. We find that explanations help people detect direct biases but not indirect biases. Additionally, regardless of bias type, explanations tend to increase agreement with model biases. Disclosures can help mitigate this effect for indirect biases, improving both unfairness recognition and the decision-making fairness. We hope that our findings can help guide further research into advancing explanations in support of fair human-AI decision-making.

{{</citation>}}


### (14/172) Transport-Hub-Aware Spatial-Temporal Adaptive Graph Transformer for Traffic Flow Prediction (Xiao Xu et al., 2023)

{{<citation>}}

Xiao Xu, Lei Zhang, Bailong Liu, Zhizhen Liang, Xuefei Zhang. (2023)  
**Transport-Hub-Aware Spatial-Temporal Adaptive Graph Transformer for Traffic Flow Prediction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.08328v1)  

---


**ABSTRACT**  
As a core technology of Intelligent Transportation System (ITS), traffic flow prediction has a wide range of applications. Traffic flow data are spatial-temporal, which are not only correlated to spatial locations in road networks, but also vary with temporal time indices. Existing methods have solved the challenges in traffic flow prediction partly, focusing on modeling spatial-temporal dependencies effectively, while not all intrinsic properties of traffic flow data are utilized fully. Besides, there are very few attempts at incremental learning of spatial-temporal data mining, and few previous works can be easily transferred to the traffic flow prediction task. Motivated by the challenge of incremental learning methods for traffic flow prediction and the underutilization of intrinsic properties of road networks, we propose a Transport-Hub-aware Spatial-Temporal adaptive graph transFormer (H-STFormer) for traffic flow prediction. Specifically, we first design a novel spatial self-attention module to capture the dynamic spatial dependencies. Three graph masking matrices are integrated into spatial self-attentions to highlight both short- and long-term dependences. Additionally, we employ a temporal self-attention module to detect dynamic temporal patterns in the traffic flow data. Finally, we design an extra spatial-temporal knowledge distillation module for incremental learning of traffic flow prediction tasks. Through extensive experiments, we show the effectiveness of H-STFormer in normal and incremental traffic flow prediction tasks. The code is available at https://github.com/Fantasy-Shaw/H-STFormer.

{{</citation>}}


### (15/172) If our aim is to build morality into an artificial agent, how might we begin to go about doing so? (Reneira Seeamber et al., 2023)

{{<citation>}}

Reneira Seeamber, Cosmin Badea. (2023)  
**If our aim is to build morality into an artificial agent, how might we begin to go about doing so?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08295v1)  

---


**ABSTRACT**  
As Artificial Intelligence (AI) becomes pervasive in most fields, from healthcare to autonomous driving, it is essential that we find successful ways of building morality into our machines, especially for decision-making. However, the question of what it means to be moral is still debated, particularly in the context of AI. In this paper, we highlight the different aspects that should be considered when building moral agents, including the most relevant moral paradigms and challenges. We also discuss the top-down and bottom-up approaches to design and the role of emotion and sentience in morality. We then propose solutions including a hybrid approach to design and a hierarchical approach to combining moral paradigms. We emphasize how governance and policy are becoming ever more critical in AI Ethics and in ensuring that the tasks we set for moral agents are attainable, that ethical behavior is achieved, and that we obtain good AI.

{{</citation>}}


### (16/172) Can Large Language Models Really Improve by Self-critiquing Their Own Plans? (Karthik Valmeekam et al., 2023)

{{<citation>}}

Karthik Valmeekam, Matthew Marquez, Subbarao Kambhampati. (2023)  
**Can Large Language Models Really Improve by Self-critiquing Their Own Plans?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08118v1)  

---


**ABSTRACT**  
There have been widespread claims about Large Language Models (LLMs) being able to successfully verify or self-critique their candidate solutions in reasoning problems in an iterative mode. Intrigued by those claims, in this paper we set out to investigate the verification/self-critiquing abilities of large language models in the context of planning. We evaluate a planning system that employs LLMs for both plan generation and verification. We assess the verifier LLM's performance against ground-truth verification, the impact of self-critiquing on plan generation, and the influence of varying feedback levels on system performance. Using GPT-4, a state-of-the-art LLM, for both generation and verification, our findings reveal that self-critiquing appears to diminish plan generation performance, especially when compared to systems with external, sound verifiers and the LLM verifiers in that system produce a notable number of false positives, compromising the system's reliability. Additionally, the nature of feedback, whether binary or detailed, showed minimal impact on plan generation. Collectively, our results cast doubt on the effectiveness of LLMs in a self-critiquing, iterative framework for planning tasks.

{{</citation>}}


### (17/172) GameGPT: Multi-agent Collaborative Framework for Game Development (Dake Chen et al., 2023)

{{<citation>}}

Dake Chen, Hanbin Wang, Yunhao Huo, Yuzhao Li, Haoyang Zhang. (2023)  
**GameGPT: Multi-agent Collaborative Framework for Game Development**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.08067v1)  

---


**ABSTRACT**  
The large language model (LLM) based agents have demonstrated their capacity to automate and expedite software development processes. In this paper, we focus on game development and propose a multi-agent collaborative framework, dubbed GameGPT, to automate game development. While many studies have pinpointed hallucination as a primary roadblock for deploying LLMs in production, we identify another concern: redundancy. Our framework presents a series of methods to mitigate both concerns. These methods include dual collaboration and layered approaches with several in-house lexicons, to mitigate the hallucination and redundancy in the planning, task identification, and implementation phases. Furthermore, a decoupling approach is also introduced to achieve code generation with better precision.

{{</citation>}}


### (18/172) Understanding and Controlling a Maze-Solving Policy Network (Ulisse Mini et al., 2023)

{{<citation>}}

Ulisse Mini, Peli Grietzer, Mrinank Sharma, Austin Meek, Monte MacDiarmid, Alexander Matt Turner. (2023)  
**Understanding and Controlling a Maze-Solving Policy Network**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08043v1)  

---


**ABSTRACT**  
To understand the goals and goal representations of AI systems, we carefully study a pretrained reinforcement learning policy that solves mazes by navigating to a range of target squares. We find this network pursues multiple context-dependent goals, and we further identify circuits within the network that correspond to one of these goals. In particular, we identified eleven channels that track the location of the goal. By modifying these channels, either with hand-designed interventions or by combining forward passes, we can partially control the policy. We show that this network contains redundant, distributed, and retargetable goal representations, shedding light on the nature of goal-direction in trained policy networks.

{{</citation>}}


### (19/172) Incorporating Domain Knowledge Graph into Multimodal Movie Genre Classification with Self-Supervised Attention and Contrastive Learning (Jiaqi Li et al., 2023)

{{<citation>}}

Jiaqi Li, Guilin Qi, Chuanyi Zhang, Yongrui Chen, Yiming Tan, Chenlong Xia, Ye Tian. (2023)  
**Incorporating Domain Knowledge Graph into Multimodal Movie Genre Classification with Self-Supervised Attention and Contrastive Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention, Contrastive Learning, Knowledge Graph, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.08032v1)  

---


**ABSTRACT**  
Multimodal movie genre classification has always been regarded as a demanding multi-label classification task due to the diversity of multimodal data such as posters, plot summaries, trailers and metadata. Although existing works have made great progress in modeling and combining each modality, they still face three issues: 1) unutilized group relations in metadata, 2) unreliable attention allocation, and 3) indiscriminative fused features. Given that the knowledge graph has been proven to contain rich information, we present a novel framework that exploits the knowledge graph from various perspectives to address the above problems. As a preparation, the metadata is processed into a domain knowledge graph. A translate model for knowledge graph embedding is adopted to capture the relations between entities. Firstly we retrieve the relevant embedding from the knowledge graph by utilizing group relations in metadata and then integrate it with other modalities. Next, we introduce an Attention Teacher module for reliable attention allocation based on self-supervised learning. It learns the distribution of the knowledge graph and produces rational attention weights. Finally, a Genre-Centroid Anchored Contrastive Learning module is proposed to strengthen the discriminative ability of fused features. The embedding space of anchors is initialized from the genre entities in the knowledge graph. To verify the effectiveness of our framework, we collect a larger and more challenging dataset named MM-IMDb 2.0 compared with the MM-IMDb dataset. The experimental results on two datasets demonstrate that our model is superior to the state-of-the-art methods. We will release the code in the near future.

{{</citation>}}


### (20/172) Effects of Human Adversarial and Affable Samples on BERT Generalizability (Aparna Elangovan et al., 2023)

{{<citation>}}

Aparna Elangovan, Jiayuan He, Yuan Li, Karin Verspoor. (2023)  
**Effects of Human Adversarial and Affable Samples on BERT Generalizability**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.08008v2)  

---


**ABSTRACT**  
BERT-based models have had strong performance on leaderboards, yet have been demonstrably worse in real-world settings requiring generalization. Limited quantities of training data is considered a key impediment to achieving generalizability in machine learning. In this paper, we examine the impact of training data quality, not quantity, on a model's generalizability. We consider two characteristics of training data: the portion of human-adversarial (h-adversarial), i.e., sample pairs with seemingly minor differences but different ground-truth labels, and human-affable (h-affable) training samples, i.e., sample pairs with minor differences but the same ground-truth label. We find that for a fixed size of training samples, as a rule of thumb, having 10-30% h-adversarial instances improves the precision, and therefore F1, by up to 20 points in the tasks of text classification and relation extraction. Increasing h-adversarials beyond this range can result in performance plateaus or even degradation. In contrast, h-affables may not contribute to a model's generalizability and may even degrade generalization performance.

{{</citation>}}


### (21/172) A Novel Statistical Measure for Out-of-Distribution Detection in Data Quality Assurance (Tinghui Ouyang et al., 2023)

{{<citation>}}

Tinghui Ouyang, Isao Echizen, Yoshiki Seo. (2023)  
**A Novel Statistical Measure for Out-of-Distribution Detection in Data Quality Assurance**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07998v1)  

---


**ABSTRACT**  
Data outside the problem domain poses significant threats to the security of AI-based intelligent systems. Aiming to investigate the data domain and out-of-distribution (OOD) data in AI quality management (AIQM) study, this paper proposes to use deep learning techniques for feature representation and develop a novel statistical measure for OOD detection. First, to extract low-dimensional representative features distinguishing normal and OOD data, the proposed research combines the deep auto-encoder (AE) architecture and neuron activation status for feature engineering. Then, using local conditional probability (LCP) in data reconstruction, a novel and superior statistical measure is developed to calculate the score of OOD detection. Experiments and evaluations are conducted on image benchmark datasets and an industrial dataset. Through comparative analysis with other common statistical measures in OOD detection, the proposed research is validated as feasible and effective in OOD and AIQM studies.

{{</citation>}}


### (22/172) Large Language Models for Scientific Synthesis, Inference and Explanation (Yizhen Zheng et al., 2023)

{{<citation>}}

Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh T. N. Nguyen, Lauren T. May, Geoffrey I. Webb, Shirui Pan. (2023)  
**Large Language Models for Scientific Synthesis, Inference and Explanation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.07984v1)  

---


**ABSTRACT**  
Large language models are a form of artificial intelligence systems whose primary knowledge consists of the statistical patterns, semantic relationships, and syntactical structures of language1. Despite their limited forms of "knowledge", these systems are adept at numerous complex tasks including creative writing, storytelling, translation, question-answering, summarization, and computer code generation. However, they have yet to demonstrate advanced applications in natural science. Here we show how large language models can perform scientific synthesis, inference, and explanation. We present a method for using general-purpose large language models to make inferences from scientific datasets of the form usually associated with special-purpose machine learning algorithms. We show that the large language model can augment this "knowledge" by synthesizing from the scientific literature. When a conventional machine learning system is augmented with this synthesized and inferred knowledge it can outperform the current state of the art across a range of benchmark tasks for predicting molecular properties. This approach has the further advantage that the large language model can explain the machine learning system's predictions. We anticipate that our framework will open new avenues for AI to accelerate the pace of scientific discovery.

{{</citation>}}


## cs.CV (31)



### (23/172) Investigating the Robustness and Properties of Detection Transformers (DETR) Toward Difficult Images (Zhao Ning Zou et al., 2023)

{{<citation>}}

Zhao Ning Zou, Yuhang Zhang, Robert Wijaya. (2023)  
**Investigating the Robustness and Properties of Detection Transformers (DETR) Toward Difficult Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08772v1)  

---


**ABSTRACT**  
Transformer-based object detectors (DETR) have shown significant performance across machine vision tasks, ultimately in object detection. This detector is based on a self-attention mechanism along with the transformer encoder-decoder architecture to capture the global context in the image. The critical issue to be addressed is how this model architecture can handle different image nuisances, such as occlusion and adversarial perturbations. We studied this issue by measuring the performance of DETR with different experiments and benchmarking the network with convolutional neural network (CNN) based detectors like YOLO and Faster-RCNN. We found that DETR performs well when it comes to resistance to interference from information loss in occlusion images. Despite that, we found that the adversarial stickers put on the image require the network to produce a new unnecessary set of keys, queries, and values, which in most cases, results in a misdirection of the network. DETR also performed poorer than YOLOv5 in the image corruption benchmark. Furthermore, we found that DETR depends heavily on the main query when making a prediction, which leads to imbalanced contributions between queries since the main query receives most of the gradient flow.

{{</citation>}}


### (24/172) Development and Validation of a Deep Learning-Based Microsatellite Instability Predictor from Prostate Cancer Whole-Slide Images (Qiyuan Hu et al., 2023)

{{<citation>}}

Qiyuan Hu, Abbas A. Rizvi, Geoffery Schau, Kshitij Ingale, Yoni Muller, Rachel Baits, Sebastian Pretzer, Aïcha BenTaieb, Abigail Gordhamer, Roberto Nussenzveig, Adam Cole, Matthew O. Leavitt, Rohan P. Joshi, Nike Beaubier, Martin C. Stumpe, Kunal Nagpal. (2023)  
**Development and Validation of a Deep Learning-Based Microsatellite Instability Predictor from Prostate Cancer Whole-Slide Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2310.08743v1)  

---


**ABSTRACT**  
Microsatellite instability-high (MSI-H) is a tumor agnostic biomarker for immune checkpoint inhibitor therapy. However, MSI status is not routinely tested in prostate cancer, in part due to low prevalence and assay cost. As such, prediction of MSI status from hematoxylin and eosin (H&E) stained whole-slide images (WSIs) could identify prostate cancer patients most likely to benefit from confirmatory testing and becoming eligible for immunotherapy. Prostate biopsies and surgical resections from de-identified records of consecutive prostate cancer patients referred to our institution were analyzed. Their MSI status was determined by next generation sequencing. Patients before a cutoff date were split into an algorithm development set (n=4015, MSI-H 1.8%) and a paired validation set (n=173, MSI-H 19.7%) that consisted of two serial sections from each sample, one stained and scanned internally and the other at an external site. Patients after the cutoff date formed the temporal validation set (n=1350, MSI-H 2.3%). Attention-based multiple instance learning models were trained to predict MSI-H from H&E WSIs. The MSI-H predictor achieved area under the receiver operating characteristic curve values of 0.78 (95% CI [0.69-0.86]), 0.72 (95% CI [0.63-0.81]), and 0.72 (95% CI [0.62-0.82]) on the internally prepared, externally prepared, and temporal validation sets, respectively. While MSI-H status is significantly correlated with Gleason score, the model remained predictive within each Gleason score subgroup. In summary, we developed and validated an AI-based MSI-H diagnostic model on a large real-world cohort of routine H&E slides, which effectively generalized to externally stained and scanned samples and a temporally independent validation cohort. This algorithm has the potential to direct prostate cancer patients toward immunotherapy and to identify MSI-H cases secondary to Lynch syndrome.

{{</citation>}}


### (25/172) Fed-Safe: Securing Federated Learning in Healthcare Against Adversarial Attacks (Erfan Darzi et al., 2023)

{{<citation>}}

Erfan Darzi, Nanna M. Sijtsema, P. M. A van Ooijen. (2023)  
**Fed-Safe: Securing Federated Learning in Healthcare Against Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.08681v1)  

---


**ABSTRACT**  
This paper explores the security aspects of federated learning applications in medical image analysis. Current robustness-oriented methods like adversarial training, secure aggregation, and homomorphic encryption often risk privacy compromises. The central aim is to defend the network against potential privacy breaches while maintaining model robustness against adversarial manipulations. We show that incorporating distributed noise, grounded in the privacy guarantees in federated settings, enables the development of a adversarially robust model that also meets federated privacy standards. We conducted comprehensive evaluations across diverse attack scenarios, parameters, and use cases in cancer imaging, concentrating on pathology, meningioma, and glioma. The results reveal that the incorporation of distributed noise allows for the attainment of security levels comparable to those of conventional adversarial training while requiring fewer retraining samples to establish a robust model.

{{</citation>}}


### (26/172) SSG2: A new modelling paradigm for semantic segmentation (Foivos I. Diakogiannis et al., 2023)

{{<citation>}}

Foivos I. Diakogiannis, Suzanne Furby, Peter Caccetta, Xiaoliang Wu, Rodrigo Ibata, Ondrej Hlinka, John Taylor. (2023)  
**SSG2: A new modelling paradigm for semantic segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.08671v1)  

---


**ABSTRACT**  
State-of-the-art models in semantic segmentation primarily operate on single, static images, generating corresponding segmentation masks. This one-shot approach leaves little room for error correction, as the models lack the capability to integrate multiple observations for enhanced accuracy. Inspired by work on semantic change detection, we address this limitation by introducing a methodology that leverages a sequence of observables generated for each static input image. By adding this "temporal" dimension, we exploit strong signal correlations between successive observations in the sequence to reduce error rates. Our framework, dubbed SSG2 (Semantic Segmentation Generation 2), employs a dual-encoder, single-decoder base network augmented with a sequence model. The base model learns to predict the set intersection, union, and difference of labels from dual-input images. Given a fixed target input image and a set of support images, the sequence model builds the predicted mask of the target by synthesizing the partial views from each sequence step and filtering out noise. We evaluate SSG2 across three diverse datasets: UrbanMonitor, featuring orthoimage tiles from Darwin, Australia with five spectral bands and 0.2m spatial resolution; ISPRS Potsdam, which includes true orthophoto images with multiple spectral bands and a 5cm ground sampling distance; and ISIC2018, a medical dataset focused on skin lesion segmentation, particularly melanoma. The SSG2 model demonstrates rapid convergence within the first few tens of epochs and significantly outperforms UNet-like baseline models with the same number of gradient updates. However, the addition of the temporal dimension results in an increased memory footprint. While this could be a limitation, it is offset by the advent of higher-memory GPUs and coding optimizations.

{{</citation>}}


### (27/172) Multimodal Large Language Model for Visual Navigation (Yao-Hung Hubert Tsai et al., 2023)

{{<citation>}}

Yao-Hung Hubert Tsai, Vansh Dhar, Jialu Li, Bowen Zhang, Jian Zhang. (2023)  
**Multimodal Large Language Model for Visual Navigation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08669v1)  

---


**ABSTRACT**  
Recent efforts to enable visual navigation using large language models have mainly focused on developing complex prompt systems. These systems incorporate instructions, observations, and history into massive text prompts, which are then combined with pre-trained large language models to facilitate visual navigation. In contrast, our approach aims to fine-tune large language models for visual navigation without extensive prompt engineering. Our design involves a simple text prompt, current observations, and a history collector model that gathers information from previous observations as input. For output, our design provides a probability distribution of possible actions that the agent can take during navigation. We train our model using human demonstrations and collision signals from the Habitat-Matterport 3D Dataset (HM3D). Experimental results demonstrate that our method outperforms state-of-the-art behavior cloning methods and effectively reduces collision rates.

{{</citation>}}


### (28/172) Octopus: Embodied Vision-Language Programmer from Environmental Feedback (Jingkang Yang et al., 2023)

{{<citation>}}

Jingkang Yang, Yuhao Dong, Shuai Liu, Bo Li, Ziyue Wang, Chencheng Jiang, Haoran Tan, Jiamu Kang, Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu. (2023)  
**Octopus: Embodied Vision-Language Programmer from Environmental Feedback**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: AI, GPT, GPT-4, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08588v1)  

---


**ABSTRACT**  
Large vision-language models (VLMs) have achieved substantial progress in multimodal perception and reasoning. Furthermore, when seamlessly integrated into an embodied agent, it signifies a crucial stride towards the creation of autonomous and context-aware systems capable of formulating plans and executing commands with precision. In this paper, we introduce Octopus, a novel VLM designed to proficiently decipher an agent's vision and textual task objectives and to formulate intricate action sequences and generate executable code. Our design allows the agent to adeptly handle a wide spectrum of tasks, ranging from mundane daily chores in simulators to sophisticated interactions in complex video games. Octopus is trained by leveraging GPT-4 to control an explorative agent to generate training data, i.e., action blueprints and the corresponding executable code, within our experimental environment called OctoVerse. We also collect the feedback that allows the enhanced training scheme of Reinforcement Learning with Environmental Feedback (RLEF). Through a series of experiments, we illuminate Octopus's functionality and present compelling results, and the proposed RLEF turns out to refine the agent's decision-making. By open-sourcing our model architecture, simulator, and dataset, we aspire to ignite further innovation and foster collaborative applications within the broader embodied AI community.

{{</citation>}}


### (29/172) PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm (Haoyi Zhu et al., 2023)

{{<citation>}}

Haoyi Zhu, Honghui Yang, Xiaoyang Wu, Di Huang, Sha Zhang, Xianglong He, Tong He, Hengshuang Zhao, Chunhua Shen, Yu Qiao, Wanli Ouyang. (2023)  
**PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.08586v2)  

---


**ABSTRACT**  
In contrast to numerous NLP and 2D computer vision foundational models, the learning of a robust and highly generalized 3D foundational model poses considerably greater challenges. This is primarily due to the inherent data variability and the diversity of downstream tasks. In this paper, we introduce a comprehensive 3D pre-training framework designed to facilitate the acquisition of efficient 3D representations, thereby establishing a pathway to 3D foundational models. Motivated by the fact that informative 3D features should be able to encode rich geometry and appearance cues that can be utilized to render realistic images, we propose a novel universal paradigm to learn point cloud representations by differentiable neural rendering, serving as a bridge between 3D and 2D worlds. We train a point cloud encoder within a devised volumetric neural renderer by comparing the rendered images with the real images. Notably, our approach demonstrates the seamless integration of the learned 3D encoder into diverse downstream tasks. These tasks encompass not only high-level challenges such as 3D detection and segmentation but also low-level objectives like 3D reconstruction and image synthesis, spanning both indoor and outdoor scenarios. Besides, we also illustrate the capability of pre-training a 2D backbone using the proposed universal methodology, surpassing conventional pre-training methods by a large margin. For the first time, PonderV2 achieves state-of-the-art performance on 11 indoor and outdoor benchmarks. The consistent improvements in various settings imply the effectiveness of the proposed method. Code and models will be made available at https://github.com/OpenGVLab/PonderV2.

{{</citation>}}


### (30/172) Is ImageNet worth 1 video? Learning strong image encoders from 1 long unlabelled video (Shashanka Venkataramanan et al., 2023)

{{<citation>}}

Shashanka Venkataramanan, Mamshad Nayeem Rizve, João Carreira, Yuki M. Asano, Yannis Avrithis. (2023)  
**Is ImageNet worth 1 video? Learning strong image encoders from 1 long unlabelled video**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08584v1)  

---


**ABSTRACT**  
Self-supervised learning has unlocked the potential of scaling up pretraining to billions of images, since annotation is unnecessary. But are we making the best use of data? How more economical can we be? In this work, we attempt to answer this question by making two contributions. First, we investigate first-person videos and introduce a "Walking Tours" dataset. These videos are high-resolution, hours-long, captured in a single uninterrupted take, depicting a large number of objects and actions with natural scene transitions. They are unlabeled and uncurated, thus realistic for self-supervision and comparable with human learning.   Second, we introduce a novel self-supervised image pretraining method tailored for learning from continuous videos. Existing methods typically adapt image-based pretraining approaches to incorporate more frames. Instead, we advocate a "tracking to learn to recognize" approach. Our method called DoRA, leads to attention maps that Discover and tRAck objects over time in an end-to-end manner, using transformer cross-attention. We derive multiple views from the tracks and use them in a classical self-supervised distillation loss. Using our novel approach, a single Walking Tours video remarkably becomes a strong competitor to ImageNet for several image and video downstream tasks.

{{</citation>}}


### (31/172) Visual Data-Type Understanding does not emerge from Scaling Vision-Language Models (Vishaal Udandarao et al., 2023)

{{<citation>}}

Vishaal Udandarao, Max F. Burg, Samuel Albanie, Matthias Bethge. (2023)  
**Visual Data-Type Understanding does not emerge from Scaling Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08577v1)  

---


**ABSTRACT**  
Recent advances in the development of vision-language models (VLMs) are yielding remarkable success in recognizing visual semantic content, including impressive instances of compositional image understanding. Here, we introduce the novel task of \textit{Visual Data-Type Identification}, a basic perceptual skill with implications for data curation (e.g., noisy data-removal from large datasets, domain-specific retrieval) and autonomous vision (e.g., distinguishing changing weather conditions from camera lens staining). We develop two datasets consisting of animal images altered across a diverse set of 27 visual \textit{data-types}, spanning four broad categories. An extensive zero-shot evaluation of 39 VLMs, ranging from 100M to 80B parameters, shows a nuanced performance landscape. While VLMs are reasonably good at identifying certain stylistic \textit{data-types}, such as cartoons and sketches, they struggle with simpler \textit{data-types} arising from basic manipulations like image rotations or additive noise. Our findings reveal that (i) model scaling alone yields marginal gains for contrastively-trained models like CLIP, and (ii) there is a pronounced drop in performance for the largest auto-regressively trained VLMs like OpenFlamingo. This finding points to a blind spot in current frontier VLMs: they excel in recognizing semantic content but fail to acquire an understanding of visual \textit{data-types} through scaling. By analyzing the pre-training distributions of these models and incorporating \textit{data-type} information into the captions during fine-tuning, we achieve a significant enhancement in performance. By exploring this previously uncharted task, we aim to set the stage for further advancing VLMs to equip them with visual data-type understanding. Code and datasets are released \href{https://github.com/bethgelab/DataTypeIdentification}{here}.

{{</citation>}}


### (32/172) Idea2Img: Iterative Self-Refinement with GPT-4V(ision) for Automatic Image Design and Generation (Zhengyuan Yang et al., 2023)

{{<citation>}}

Zhengyuan Yang, Jianfeng Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Lijuan Wang. (2023)  
**Idea2Img: Iterative Self-Refinement with GPT-4V(ision) for Automatic Image Design and Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.08541v1)  

---


**ABSTRACT**  
We introduce ``Idea to Image,'' a system that enables multimodal iterative self-refinement with GPT-4V(ision) for automatic image design and generation. Humans can quickly identify the characteristics of different text-to-image (T2I) models via iterative explorations. This enables them to efficiently convert their high-level generation ideas into effective T2I prompts that can produce good images. We investigate if systems based on large multimodal models (LMMs) can develop analogous multimodal self-refinement abilities that enable exploring unknown models or environments via self-refining tries. Idea2Img cyclically generates revised T2I prompts to synthesize draft images, and provides directional feedback for prompt revision, both conditioned on its memory of the probed T2I model's characteristics. The iterative self-refinement brings Idea2Img various advantages over vanilla T2I models. Notably, Idea2Img can process input ideas with interleaved image-text sequences, follow ideas with design instructions, and generate images of better semantic and visual qualities. The user preference study validates the efficacy of multimodal iterative self-refinement on automatic image design and generation.

{{</citation>}}


### (33/172) XAI Benchmark for Visual Explanation (Yifei Zhang et al., 2023)

{{<citation>}}

Yifei Zhang, Siyi Gu, James Song, Bo Pan, Liang Zhao. (2023)  
**XAI Benchmark for Visual Explanation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08537v1)  

---


**ABSTRACT**  
The rise of deep learning algorithms has led to significant advancements in computer vision tasks, but their "black box" nature has raised concerns regarding interpretability. Explainable AI (XAI) has emerged as a critical area of research aiming to open this "black box", and shed light on the decision-making process of AI models. Visual explanations, as a subset of Explainable Artificial Intelligence (XAI), provide intuitive insights into the decision-making processes of AI models handling visual data by highlighting influential areas in an input image. Despite extensive research conducted on visual explanations, most evaluations are model-centered since the availability of corresponding real-world datasets with ground truth explanations is scarce in the context of image data. To bridge this gap, we introduce an XAI Benchmark comprising a dataset collection from diverse topics that provide both class labels and corresponding explanation annotations for images. We have processed data from diverse domains to align with our unified visual explanation framework. We introduce a comprehensive Visual Explanation pipeline, which integrates data loading, preprocessing, experimental setup, and model evaluation processes. This structure enables researchers to conduct fair comparisons of various visual explanation techniques. In addition, we provide a comprehensive review of over 10 evaluation methods for visual explanation to assist researchers in effectively utilizing our dataset collection. To further assess the performance of existing visual explanation methods, we conduct experiments on selected datasets using various model-centered and ground truth-centered evaluation metrics. We envision this benchmark could facilitate the advancement of visual explanation models. The XAI dataset collection and easy-to-use code for evaluation are publicly accessible at https://xaidataset.github.io.

{{</citation>}}


### (34/172) Assessing of Soil Erosion Risk Through Geoinformation Sciences and Remote Sensing -- A Review (Lachezar Filchev et al., 2023)

{{<citation>}}

Lachezar Filchev, Vasil Kolev. (2023)  
**Assessing of Soil Erosion Risk Through Geoinformation Sciences and Remote Sensing -- A Review**  

---
Primary Category: cs.CV  
Categories: 74Lxx, 91B05, 86-01, H-4; J-2, cond-mat-dis-nn, cs-CV, cs.CV, physics-data-an, physics-geo-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08430v1)  

---


**ABSTRACT**  
During past decades a marked manifestation of widespread erosion phenomena was studied worldwide. Global conservation community has launched campaigns at local, regional and continental level in developing countries for preservation of soil resources in order not only to stop or mitigate human impact on nature but also to improve life in rural areas introducing new approaches for soil cultivation. After the adoption of Sustainable Development Goals of UNs and launching several world initiatives such as the Land Degradation Neutrality (LDN) the world came to realize the very importance of the soil resources on which the biosphere relies for its existence. The main goal of the chapter is to review different types and structures erosion models as well as their applications. Several methods using spatial analysis capabilities of geographic information systems (GIS) are in operation for soil erosion risk assessment, such as Universal Soil Loss Equation (USLE), Revised Universal Soil Loss Equation (RUSLE) in operation worldwide and in the USA and MESALES model. These and more models are being discussed in the present work alongside more experimental models and methods for assessing soil erosion risk such as Artificial Intelligence (AI), Machine and Deep Learning, etc. At the end of this work, a prospectus for the future development of soil erosion risk assessment is drawn.

{{</citation>}}


### (35/172) Revisiting Data Augmentation for Rotational Invariance in Convolutional Neural Networks (Facundo Manuel Quiroga et al., 2023)

{{<citation>}}

Facundo Manuel Quiroga, Franco Ronchetti, Laura Lanzarini, Aurelio Fernandez-Bariviera. (2023)  
**Revisiting Data Augmentation for Rotational Invariance in Convolutional Neural Networks**  

---
Primary Category: cs.CV  
Categories: I-2-10; I-5-2, cs-CV, cs-NE, cs.CV  
Keywords: Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08429v1)  

---


**ABSTRACT**  
Convolutional Neural Networks (CNN) offer state of the art performance in various computer vision tasks. Many of those tasks require different subtypes of affine invariances (scale, rotational, translational) to image transformations. Convolutional layers are translation equivariant by design, but in their basic form lack invariances. In this work we investigate how best to include rotational invariance in a CNN for image classification. Our experiments show that networks trained with data augmentation alone can classify rotated images nearly as well as in the normal unrotated case; this increase in representational power comes only at the cost of training time. We also compare data augmentation versus two modified CNN models for achieving rotational invariance or equivariance, Spatial Transformer Networks and Group Equivariant CNNs, finding no significant accuracy increase with these specialized methods. In the case of data augmented networks, we also analyze which layers help the network to encode the rotational invariance, which is important for understanding its limitations and how to best retrain a network with data augmentation to achieve invariance to rotation.

{{</citation>}}


### (36/172) 'SegLoc': Study on Novel Visual Self-supervised Learning Scheme (Segment Localization) Tailored for Dense Prediction Tasks of Security Inspection X-ray Images (Shervin Halat et al., 2023)

{{<citation>}}

Shervin Halat, Mohammad Rahmati, Ehsan Nazerfard. (2023)  
**'SegLoc': Study on Novel Visual Self-supervised Learning Scheme (Segment Localization) Tailored for Dense Prediction Tasks of Security Inspection X-ray Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, NLP, Security  
[Paper Link](http://arxiv.org/abs/2310.08421v1)  

---


**ABSTRACT**  
Lately, remarkable advancements of artificial intelligence have been attributed to the integration of self-supervised learning scheme. Despite impressive achievements within NLP, yet SSL in computer vision has not been able to stay on track comparatively. Recently, integration of contrastive learning on top of existing SSL models has established considerable progress in computer vision through which visual SSL models have outperformed their supervised counterparts. Nevertheless, most of these improvements were limited to classification tasks, and also, few works have been dedicated to evaluation of SSL models in real-world scenarios of computer vision, while the majority of works are centered around datasets containing class-wise portrait images, most notably, ImageNet. Consequently, in this work, we have considered dense prediction task of semantic segmentation in security inspection x-ray images to evaluate our proposed model Segmentation Localization. Based upon the model Instance Localization, our model SegLoc has managed to address one of the most challenging downsides of contrastive learning, i.e., false negative pairs of query embeddings. In order to do so, in contrast to baseline model InsLoc, our pretraining dataset is synthesized by cropping, transforming, then pasting already labeled segments from an available labeled dataset, foregrounds, onto instances of an unlabeled dataset, backgrounds. In our case, PIDray and SIXray datasets are considered as labeled and unlabeled datasets, respectively. Moreover, we fully harness labels by avoiding false negative pairs through implementing the idea, one queue per class, in MoCo-v2 whereby negative pairs corresponding to each query are extracted from its corresponding queue within the memory bank. Our approach has outperformed random initialization by 3% to 6%, while having underperformed supervised initialization.

{{</citation>}}


### (37/172) Visual Attention-Prompted Prediction and Learning (Yifei Zhang et al., 2023)

{{<citation>}}

Yifei Zhang, Siyi Gu, Bo Pan, Guangji Bai, Xiaofeng Yang, Liang Zhao. (2023)  
**Visual Attention-Prompted Prediction and Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.08420v1)  

---


**ABSTRACT**  
Explanation(attention)-guided learning is a method that enhances a model's predictive power by incorporating human understanding during the training phase. While attention-guided learning has shown promising results, it often involves time-consuming and computationally expensive model retraining. To address this issue, we introduce the attention-prompted prediction technique, which enables direct prediction guided by the attention prompt without the need for model retraining. However, this approach presents several challenges, including: 1) How to incorporate the visual attention prompt into the model's decision-making process and leverage it for future predictions even in the absence of a prompt? and 2) How to handle the incomplete information from the visual attention prompt? To tackle these challenges, we propose a novel framework called Visual Attention-Prompted Prediction and Learning, which seamlessly integrates visual attention prompts into the model's decision-making process and adapts to images both with and without attention prompts for prediction. To address the incomplete information of the visual attention prompt, we introduce a perturbation-based attention map modification method. Additionally, we propose an optimization-based mask aggregation method with a new weight learning function for adaptive perturbed annotation aggregation in the attention map modification process. Our overall framework is designed to learn in an attention-prompt guided multi-task manner to enhance future predictions even for samples without attention prompts and trained in an alternating manner for better convergence. Extensive experiments conducted on two datasets demonstrate the effectiveness of our proposed framework in enhancing predictions for samples, both with and without provided prompts.

{{</citation>}}


### (38/172) MeanAP-Guided Reinforced Active Learning for Object Detection (Zhixuan Liang et al., 2023)

{{<citation>}}

Zhixuan Liang, Xingyu Zeng, Rui Zhao, Ping Luo. (2023)  
**MeanAP-Guided Reinforced Active Learning for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Active Learning, LSTM, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.08387v1)  

---


**ABSTRACT**  
Active learning presents a promising avenue for training high-performance models with minimal labeled data, achieved by judiciously selecting the most informative instances to label and incorporating them into the task learner. Despite notable advancements in active learning for image recognition, metrics devised or learned to gauge the information gain of data, crucial for query strategy design, do not consistently align with task model performance metrics, such as Mean Average Precision (MeanAP) in object detection tasks. This paper introduces MeanAP-Guided Reinforced Active Learning for Object Detection (MAGRAL), a novel approach that directly utilizes the MeanAP metric of the task model to devise a sampling strategy employing a reinforcement learning-based sampling agent. Built upon LSTM architecture, the agent efficiently explores and selects subsequent training instances, and optimizes the process through policy gradient with MeanAP serving as reward. Recognizing the time-intensive nature of MeanAP computation at each step, we propose fast look-up tables to expedite agent training. We assess MAGRAL's efficacy across popular benchmarks, PASCAL VOC and MS COCO, utilizing different backbone architectures. Empirical findings substantiate MAGRAL's superiority over recent state-of-the-art methods, showcasing substantial performance gains. MAGRAL establishes a robust baseline for reinforced active object detection, signifying its potential in advancing the field.

{{</citation>}}


### (39/172) CHIP: Contrastive Hierarchical Image Pretraining (Arpit Mittal et al., 2023)

{{<citation>}}

Arpit Mittal, Harshil Jhaveri, Swapnil Mallick, Abhishek Ajmera. (2023)  
**CHIP: Contrastive Hierarchical Image Pretraining**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08304v1)  

---


**ABSTRACT**  
Few-shot object classification is the task of classifying objects in an image with limited number of examples as supervision. We propose a one-shot/few-shot classification model that can classify an object of any unseen class into a relatively general category in an hierarchically based classification. Our model uses a three-level hierarchical contrastive loss based ResNet152 classifier for classifying an object based on its features extracted from Image embedding, not used during the training phase. For our experimentation, we have used a subset of the ImageNet (ILSVRC-12) dataset that contains only the animal classes for training our model and created our own dataset of unseen classes for evaluating our trained model. Our model provides satisfactory results in classifying the unknown objects into a generic category which has been later discussed in greater detail.

{{</citation>}}


### (40/172) Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-text Retrieval (Qing Ma et al., 2023)

{{<citation>}}

Qing Ma, Jiancheng Pan, Cong Bai. (2023)  
**Direction-Oriented Visual-semantic Embedding Model for Remote Sensing Image-text Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2310.08276v1)  

---


**ABSTRACT**  
Image-text retrieval has developed rapidly in recent years. However, it is still a challenge in remote sensing due to visual-semantic imbalance, which leads to incorrect matching of non-semantic visual and textual features. To solve this problem, we propose a novel Direction-Oriented Visual-semantic Embedding Model (DOVE) to mine the relationship between vision and language. Concretely, a Regional-Oriented Attention Module (ROAM) adaptively adjusts the distance between the final visual and textual embeddings in the latent semantic space, oriented by regional visual features. Meanwhile, a lightweight Digging Text Genome Assistant (DTGA) is designed to expand the range of tractable textual representation and enhance global word-level semantic connections using less attention operations. Ultimately, we exploit a global visual-semantic constraint to reduce single visual dependency and serve as an external constraint for the final visual and textual representations. The effectiveness and superiority of our method are verified by extensive experiments including parameter evaluation, quantitative comparison, ablation studies and visual analysis, on two benchmark datasets, RSICD and RSITMD.

{{</citation>}}


### (41/172) GraphAlign: Enhancing Accurate Feature Alignment by Graph matching for Multi-Modal 3D Object Detection (Ziying Song et al., 2023)

{{<citation>}}

Ziying Song, Haiyue Wei, Lin Bai, Lei Yang, Caiyan Jia. (2023)  
**GraphAlign: Enhancing Accurate Feature Alignment by Graph matching for Multi-Modal 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.08261v1)  

---


**ABSTRACT**  
LiDAR and cameras are complementary sensors for 3D object detection in autonomous driving. However, it is challenging to explore the unnatural interaction between point clouds and images, and the critical factor is how to conduct feature alignment of heterogeneous modalities. Currently, many methods achieve feature alignment by projection calibration only, without considering the problem of coordinate conversion accuracy errors between sensors, leading to sub-optimal performance. In this paper, we present GraphAlign, a more accurate feature alignment strategy for 3D object detection by graph matching. Specifically, we fuse image features from a semantic segmentation encoder in the image branch and point cloud features from a 3D Sparse CNN in the LiDAR branch. To save computation, we construct the nearest neighbor relationship by calculating Euclidean distance within the subspaces that are divided into the point cloud features. Through the projection calibration between the image and point cloud, we project the nearest neighbors of point cloud features onto the image features. Then by matching the nearest neighbors with a single point cloud to multiple images, we search for a more appropriate feature alignment. In addition, we provide a self-attention module to enhance the weights of significant relations to fine-tune the feature alignment between heterogeneous modalities. Extensive experiments on nuScenes benchmark demonstrate the effectiveness and efficiency of our GraphAlign.

{{</citation>}}


### (42/172) Distilling from Vision-Language Models for Improved OOD Generalization in Vision Tasks (Sravanti Addepalli et al., 2023)

{{<citation>}}

Sravanti Addepalli, Ashish Ramayee Asokan, Lakshay Sharma, R. Venkatesh Babu. (2023)  
**Distilling from Vision-Language Models for Improved OOD Generalization in Vision Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08255v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs) such as CLIP are trained on large amounts of image-text pairs, resulting in remarkable generalization across several data distributions. The prohibitively expensive training and data collection/curation costs of these models make them valuable Intellectual Property (IP) for organizations. This motivates a vendor-client paradigm, where a vendor trains a large-scale VLM and grants only input-output access to clients on a pay-per-query basis in a black-box setting. The client aims to minimize inference cost by distilling the VLM to a student model using the limited available task-specific data, and further deploying this student model in the downstream application. While naive distillation largely improves the In-Domain (ID) accuracy of the student, it fails to transfer the superior out-of-distribution (OOD) generalization of the VLM teacher using the limited available labeled images. To mitigate this, we propose Vision-Language to Vision-Align, Distill, Predict (VL2V-ADiP), which first aligns the vision and language modalities of the teacher model with the vision modality of a pre-trained student model, and further distills the aligned VLM embeddings to the student. This maximally retains the pre-trained features of the student, while also incorporating the rich representations of the VLM image encoder and the superior generalization of the text embeddings. The proposed approach achieves state-of-the-art results on the standard Domain Generalization benchmarks in a black-box teacher setting, and also when weights of the VLM are accessible.

{{</citation>}}


### (43/172) Long-Tailed Classification Based on Coarse-Grained Leading Forest and Multi-Center Loss (Jinye Yang et al., 2023)

{{<citation>}}

Jinye Yang, Ji Xu. (2023)  
**Long-Tailed Classification Based on Coarse-Grained Leading Forest and Multi-Center Loss**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08206v1)  

---


**ABSTRACT**  
Long-tailed(LT) classification is an unavoidable and challenging problem in the real world. Most of the existing long-tailed classification methods focus only on solving the inter-class imbalance in which there are more samples in the head class than in the tail class, while ignoring the intra-lass imbalance in which the number of samples of the head attribute within the same class is much larger than the number of samples of the tail attribute. The deviation in the model is caused by both of these factors, and due to the fact that attributes are implicit in most datasets and the combination of attributes is very complex, the intra-class imbalance is more difficult to handle. For this purpose, we proposed a long-tailed classification framework, known as \textbf{\textsc{Cognisance}}, which is founded on Coarse-Grained Leading Forest (CLF) and Multi-Center Loss (MCL), aiming to build a multi-granularity joint solution model by means of invariant feature learning. In this method, we designed an unsupervised learning method, i.e., CLF, to better characterize the distribution of attributes within a class. Depending on the distribution of attributes, we can flexibly construct sampling strategies suitable for different environments. In addition, we introduce a new metric learning loss (MCL), which aims to gradually eliminate confusing attributes during the feature learning process. More importantly, this approach does not depend on a specific model structure and can be integrated with existing LT methods as an independent component. We have conducted extensive experiments and our approach has state-of-the-art performance in both existing benchmarks ImageNet-GLT and MSCOCO-GLT, and can improve the performance of existing LT methods. Our codes are available on GitHub: \url{https://github.com/jinyery/cognisance}

{{</citation>}}


### (44/172) XIMAGENET-12: An Explainable AI Benchmark Dataset for Model Robustness Evaluation (Qiang Li et al., 2023)

{{<citation>}}

Qiang Li, Dan Zhang, Shengzhao Lei, Xun Zhao, Shuyan Li, Porawit Kamnoedboon, WeiWei Li. (2023)  
**XIMAGENET-12: An Explainable AI Benchmark Dataset for Model Robustness Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08182v1)  

---


**ABSTRACT**  
The lack of standardized robustness metrics and the widespread reliance on numerous unrelated benchmark datasets for testing have created a gap between academically validated robust models and their often problematic practical adoption. To address this, we introduce XIMAGENET-12, an explainable benchmark dataset with over 200K images and 15,600 manual semantic annotations. Covering 12 categories from ImageNet to represent objects commonly encountered in practical life and simulating six diverse scenarios, including overexposure, blurring, color changing, etc., we further propose a novel robustness criterion that extends beyond model generation ability assessment. This benchmark dataset, along with related code, is available at https://sites.google.com/view/ximagenet-12/home. Researchers and practitioners can leverage this resource to evaluate the robustness of their visual models under challenging conditions and ultimately benefit from the demands of practical computer vision systems.

{{</citation>}}


### (45/172) Fine-Grained Annotation for Face Anti-Spoofing (Xu Chen et al., 2023)

{{<citation>}}

Xu Chen, Yunde Jia, Yuwei Wu. (2023)  
**Fine-Grained Annotation for Face Anti-Spoofing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.08142v1)  

---


**ABSTRACT**  
Face anti-spoofing plays a critical role in safeguarding facial recognition systems against presentation attacks. While existing deep learning methods show promising results, they still suffer from the lack of fine-grained annotations, which lead models to learn task-irrelevant or unfaithful features. In this paper, we propose a fine-grained annotation method for face anti-spoofing. Specifically, we first leverage the Segment Anything Model (SAM) to obtain pixel-wise segmentation masks by utilizing face landmarks as point prompts. The face landmarks provide segmentation semantics, which segments the face into regions. We then adopt these regions as masks and assemble them into three separate annotation maps: spoof, living, and background maps. Finally, we combine three separate maps into a three-channel map as annotations for model training. Furthermore, we introduce the Multi-Channel Region Exchange Augmentation (MCREA) to diversify training data and reduce overfitting. Experimental results demonstrate that our method outperforms existing state-of-the-art approaches in both intra-dataset and cross-dataset evaluations.

{{</citation>}}


### (46/172) DualAug: Exploiting Additional Heavy Augmentation with OOD Data Rejection (Zehao Wang et al., 2023)

{{<citation>}}

Zehao Wang, Yiwen Guo, Qizhang Li, Guanglei Yang, Wangmeng Zuo. (2023)  
**DualAug: Exploiting Additional Heavy Augmentation with OOD Data Rejection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.08139v1)  

---


**ABSTRACT**  
Data augmentation is a dominant method for reducing model overfitting and improving generalization. Most existing data augmentation methods tend to find a compromise in augmenting the data, \textit{i.e.}, increasing the amplitude of augmentation carefully to avoid degrading some data too much and doing harm to the model performance. We delve into the relationship between data augmentation and model performance, revealing that the performance drop with heavy augmentation comes from the presence of out-of-distribution (OOD) data. Nonetheless, as the same data transformation has different effects for different training samples, even for heavy augmentation, there remains part of in-distribution data which is beneficial to model training. Based on the observation, we propose a novel data augmentation method, named \textbf{DualAug}, to keep the augmentation in distribution as much as possible at a reasonable time and computational cost. We design a data mixing strategy to fuse augmented data from both the basic- and the heavy-augmentation branches. Extensive experiments on supervised image classification benchmarks show that DualAug improve various automated data augmentation method. Moreover, the experiments on semi-supervised learning and contrastive self-supervised learning demonstrate that our DualAug can also improve related method. Code is available at \href{https://github.com/shuguang99/DualAug}{https://github.com/shuguang99/DualAug}.

{{</citation>}}


### (47/172) DUSA: Decoupled Unsupervised Sim2Real Adaptation for Vehicle-to-Everything Collaborative Perception (Xianghao Kong et al., 2023)

{{<citation>}}

Xianghao Kong, Wentao Jiang, Jinrang Jia, Yifeng Shi, Runsheng Xu, Si Liu. (2023)  
**DUSA: Decoupled Unsupervised Sim2Real Adaptation for Vehicle-to-Everything Collaborative Perception**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08117v1)  

---


**ABSTRACT**  
Vehicle-to-Everything (V2X) collaborative perception is crucial for autonomous driving. However, achieving high-precision V2X perception requires a significant amount of annotated real-world data, which can always be expensive and hard to acquire. Simulated data have raised much attention since they can be massively produced at an extremely low cost. Nevertheless, the significant domain gap between simulated and real-world data, including differences in sensor type, reflectance patterns, and road surroundings, often leads to poor performance of models trained on simulated data when evaluated on real-world data. In addition, there remains a domain gap between real-world collaborative agents, e.g. different types of sensors may be installed on autonomous vehicles and roadside infrastructures with different extrinsics, further increasing the difficulty of sim2real generalization. To take full advantage of simulated data, we present a new unsupervised sim2real domain adaptation method for V2X collaborative detection named Decoupled Unsupervised Sim2Real Adaptation (DUSA). Our new method decouples the V2X collaborative sim2real domain adaptation problem into two sub-problems: sim2real adaptation and inter-agent adaptation. For sim2real adaptation, we design a Location-adaptive Sim2Real Adapter (LSA) module to adaptively aggregate features from critical locations of the feature map and align the features between simulated data and real-world data via a sim/real discriminator on the aggregated global feature. For inter-agent adaptation, we further devise a Confidence-aware Inter-agent Adapter (CIA) module to align the fine-grained features from heterogeneous agents under the guidance of agent-wise confidence maps. Experiments demonstrate the effectiveness of the proposed DUSA approach on unsupervised sim2real adaptation from the simulated V2XSet dataset to the real-world DAIR-V2X-C dataset.

{{</citation>}}


### (48/172) Generalized Logit Adjustment: Calibrating Fine-tuned Models by Removing Label Bias in Foundation Models (Beier Zhu et al., 2023)

{{<citation>}}

Beier Zhu, Kaihua Tang, Qianru Sun, Hanwang Zhang. (2023)  
**Generalized Logit Adjustment: Calibrating Fine-tuned Models by Removing Label Bias in Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08106v1)  

---


**ABSTRACT**  
Foundation models like CLIP allow zero-shot transfer on various tasks without additional training data. Yet, the zero-shot performance is less competitive than a fully supervised one. Thus, to enhance the performance, fine-tuning and ensembling are also commonly adopted to better fit the downstream tasks. However, we argue that such prior work has overlooked the inherent biases in foundation models. Due to the highly imbalanced Web-scale training set, these foundation models are inevitably skewed toward frequent semantics, and thus the subsequent fine-tuning or ensembling is still biased. In this study, we systematically examine the biases in foundation models and demonstrate the efficacy of our proposed Generalized Logit Adjustment (GLA) method. Note that bias estimation in foundation models is challenging, as most pre-train data cannot be explicitly accessed like in traditional long-tailed classification tasks. To this end, GLA has an optimization-based bias estimation approach for debiasing foundation models. As our work resolves a fundamental flaw in the pre-training, the proposed GLA demonstrates significant improvements across a diverse range of tasks: it achieves 1.5 pp accuracy gains on ImageNet, an large average improvement (1.4-4.6 pp) on 11 few-shot datasets, 2.4 pp gains on long-tailed classification. Codes are in \url{https://github.com/BeierZhu/GLA}.

{{</citation>}}


### (49/172) Implicit Shape and Appearance Priors for Few-Shot Full Head Reconstruction (Pol Caselles et al., 2023)

{{<citation>}}

Pol Caselles, Eduard Ramon, Jaime Garcia, Gil Triginer, Francesc Moreno-Noguer. (2023)  
**Implicit Shape and Appearance Priors for Few-Shot Full Head Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.08784v1)  

---


**ABSTRACT**  
Recent advancements in learning techniques that employ coordinate-based neural representations have yielded remarkable results in multi-view 3D reconstruction tasks. However, these approaches often require a substantial number of input views (typically several tens) and computationally intensive optimization procedures to achieve their effectiveness. In this paper, we address these limitations specifically for the problem of few-shot full 3D head reconstruction. We accomplish this by incorporating a probabilistic shape and appearance prior into coordinate-based representations, enabling faster convergence and improved generalization when working with only a few input images (even as low as a single image). During testing, we leverage this prior to guide the fitting process of a signed distance function using a differentiable renderer. By incorporating the statistical prior alongside parallelizable ray tracing and dynamic caching strategies, we achieve an efficient and accurate approach to few-shot full 3D head reconstruction. Moreover, we extend the H3DS dataset, which now comprises 60 high-resolution 3D full head scans and their corresponding posed images and masks, which we use for evaluation purposes. By leveraging this dataset, we demonstrate the remarkable capabilities of our approach in achieving state-of-the-art results in geometry reconstruction while being an order of magnitude faster than previous approaches.

{{</citation>}}


### (50/172) Age Estimation Based on Graph Convolutional Networks and Multi-head Attention Mechanisms (Miaomiao Yang et al., 2023)

{{<citation>}}

Miaomiao Yang, Changwei Yao, Shijin Yan. (2023)  
**Age Estimation Based on Graph Convolutional Networks and Multi-head Attention Mechanisms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Graph Convolutional Network, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08064v1)  

---


**ABSTRACT**  
Age estimation technology is a part of facial recognition and has been applied to identity authentication. This technology achieves the development and application of a juvenile anti-addiction system by authenticating users in the game. Convolutional Neural Network (CNN) and Transformer algorithms are widely used in this application scenario. However, these two models cannot flexibly extract and model features of faces with irregular shapes, and they are ineffective in capturing key information. Furthermore, the above methods will contain a lot of background information while extracting features, which will interfere with the model. In consequence, it is easy to extract redundant information from images. In this paper, a new modeling idea is proposed to solve this problem, which can flexibly model irregular objects. The Graph Convolutional Network (GCN) is used to extract features from irregular face images effectively, and multi-head attention mechanisms are added to avoid redundant features and capture key region information in the image. This model can effectively improve the accuracy of age estimation and reduce the MAE error value to about 3.64, which is better than the effect of today's age estimation model, to improve the accuracy of face recognition and identity authentication.

{{</citation>}}


### (51/172) X-HRNet: Towards Lightweight Human Pose Estimation with Spatially Unidimensional Self-Attention (Yixuan Zhou et al., 2023)

{{<citation>}}

Yixuan Zhou, Xuanhan Wang, Xing Xu, Lei Zhao, Jingkuan Song. (2023)  
**X-HRNet: Towards Lightweight Human Pose Estimation with Spatially Unidimensional Self-Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2310.08042v1)  

---


**ABSTRACT**  
High-resolution representation is necessary for human pose estimation to achieve high performance, and the ensuing problem is high computational complexity. In particular, predominant pose estimation methods estimate human joints by 2D single-peak heatmaps. Each 2D heatmap can be horizontally and vertically projected to and reconstructed by a pair of 1D heat vectors. Inspired by this observation, we introduce a lightweight and powerful alternative, Spatially Unidimensional Self-Attention (SUSA), to the pointwise (1x1) convolution that is the main computational bottleneck in the depthwise separable 3c3 convolution. Our SUSA reduces the computational complexity of the pointwise (1x1) convolution by 96% without sacrificing accuracy. Furthermore, we use the SUSA as the main module to build our lightweight pose estimation backbone X-HRNet, where `X' represents the estimated cross-shape attention vectors. Extensive experiments on the COCO benchmark demonstrate the superiority of our X-HRNet, and comprehensive ablation studies show the effectiveness of the SUSA modules. The code is publicly available at https://github.com/cool-xuan/x-hrnet.

{{</citation>}}


### (52/172) BaSAL: Size Balanced Warm Start Active Learning for LiDAR Semantic Segmentation (Jiarong Wei et al., 2023)

{{<citation>}}

Jiarong Wei, Yancong Lin, Holger Caesar. (2023)  
**BaSAL: Size Balanced Warm Start Active Learning for LiDAR Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.08035v1)  

---


**ABSTRACT**  
Active learning strives to reduce the need for costly data annotation, by repeatedly querying an annotator to label the most informative samples from a pool of unlabeled data and retraining a model from these samples. We identify two problems with existing active learning methods for LiDAR semantic segmentation. First, they ignore the severe class imbalance inherent in LiDAR semantic segmentation datasets. Second, to bootstrap the active learning loop, they train their initial model from randomly selected data samples, which leads to low performance and is referred to as the cold start problem. To address these problems we propose BaSAL, a size-balanced warm start active learning model, based on the observation that each object class has a characteristic size. By sampling object clusters according to their size, we can thus create a size-balanced dataset that is also more class-balanced. Furthermore, in contrast to existing information measures like entropy or CoreSet, size-based sampling does not require an already trained model and thus can be used to address the cold start problem. Results show that we are able to improve the performance of the initial model by a large margin. Combining size-balanced sampling and warm start with established information measures, our approach achieves a comparable performance to training on the entire SemanticKITTI dataset, despite using only 5% of the annotations, which outperforms existing active learning methods. We also match the existing state-of-the-art in active learning on nuScenes. Our code will be made available upon paper acceptance.

{{</citation>}}


### (53/172) Self-supervised visual learning for analyzing firearms trafficking activities on the Web (Sotirios Konstantakos et al., 2023)

{{<citation>}}

Sotirios Konstantakos, Despina Ioanna Chalkiadaki, Ioannis Mademlis, Adamantia Anna Rebolledo Chrysochoou, Georgios Th. Papadopoulos. (2023)  
**Self-supervised visual learning for analyzing firearms trafficking activities on the Web**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07975v1)  

---


**ABSTRACT**  
Automated visual firearms classification from RGB images is an important real-world task with applications in public space security, intelligence gathering and law enforcement investigations. When applied to images massively crawled from the World Wide Web (including social media and dark Web sites), it can serve as an important component of systems that attempt to identify criminal firearms trafficking networks, by analyzing Big Data from open-source intelligence. Deep Neural Networks (DNN) are the state-of-the-art methodology for achieving this, with Convolutional Neural Networks (CNN) being typically employed. The common transfer learning approach consists of pretraining on a large-scale, generic annotated dataset for whole-image classification, such as ImageNet-1k, and then finetuning the DNN on a smaller, annotated, task-specific, downstream dataset for visual firearms classification. Neither Visual Transformer (ViT) neural architectures nor Self-Supervised Learning (SSL) approaches have been so far evaluated on this critical task. SSL essentially consists of replacing the traditional supervised pretraining objective with an unsupervised pretext task that does not require ground-truth labels..

{{</citation>}}


## cond-mat.mes-hall (1)



### (54/172) Modeling Fission Gas Release at the Mesoscale using Multiscale DenseNet Regression with Attention Mechanism and Inception Blocks (Peter Toma et al., 2023)

{{<citation>}}

Peter Toma, Md Ali Muntaha, Joel B. Harley, Michael R. Tonks. (2023)  
**Modeling Fission Gas Release at the Mesoscale using Multiscale DenseNet Regression with Attention Mechanism and Inception Blocks**  

---
Primary Category: cond-mat.mes-hall  
Categories: cond-mat-dis-nn, cond-mat-mes-hall, cond-mat.mes-hall, cs-LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.08767v1)  

---


**ABSTRACT**  
Mesoscale simulations of fission gas release (FGR) in nuclear fuel provide a powerful tool for understanding how microstructure evolution impacts FGR, but they are computationally intensive. In this study, we present an alternate, data-driven approach, using deep learning to predict instantaneous FGR flux from 2D nuclear fuel microstructure images. Four convolutional neural network (CNN) architectures with multiscale regression are trained and evaluated on simulated FGR data generated using a hybrid phase field/cluster dynamics model. All four networks show high predictive power, with $R^{2}$ values above 98%. The best performing network combine a Convolutional Block Attention Module (CBAM) and InceptionNet mechanisms to provide superior accuracy (mean absolute percentage error of 4.4%), training stability, and robustness on very low instantaneous FGR flux values.

{{</citation>}}


## cs.CL (47)



### (55/172) Calibrating Likelihoods towards Consistency in Summarization Models (Polina Zablotskaia et al., 2023)

{{<citation>}}

Polina Zablotskaia, Misha Khalman, Rishabh Joshi, Livio Baldini Soares, Shoshana Jakobovits, Joshua Maynez, Shashi Narayan. (2023)  
**Calibrating Likelihoods towards Consistency in Summarization Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLI, Summarization  
[Paper Link](http://arxiv.org/abs/2310.08764v1)  

---


**ABSTRACT**  
Despite the recent advances in abstractive text summarization, current summarization models still suffer from generating factually inconsistent summaries, reducing their utility for real-world application. We argue that the main reason for such behavior is that the summarization models trained with maximum likelihood objective assign high probability to plausible sequences given the context, but they often do not accurately rank sequences by their consistency. In this work, we solve this problem by calibrating the likelihood of model generated sequences to better align with a consistency metric measured by natural language inference (NLI) models. The human evaluation study and automatic metrics show that the calibrated models generate more consistent and higher-quality summaries. We also show that the models trained using our method return probabilities that are better aligned with the NLI scores, which significantly increase reliability of summarization models.

{{</citation>}}


### (56/172) Circuit Component Reuse Across Tasks in Transformer Language Models (Jack Merullo et al., 2023)

{{<citation>}}

Jack Merullo, Carsten Eickhoff, Ellie Pavlick. (2023)  
**Circuit Component Reuse Across Tasks in Transformer Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08744v1)  

---


**ABSTRACT**  
Recent work in mechanistic interpretability has shown that behaviors in language models can be successfully reverse-engineered through circuit analysis. A common criticism, however, is that each circuit is task-specific, and thus such analysis cannot contribute to understanding the models at a higher level. In this work, we present evidence that insights (both low-level findings about specific heads and higher-level findings about general algorithms) can indeed generalize across tasks. Specifically, we study the circuit discovered in Wang et al. (2022) for the Indirect Object Identification (IOI) task and 1.) show that it reproduces on a larger GPT2 model, and 2.) that it is mostly reused to solve a seemingly different task: Colored Objects (Ippolito & Callison-Burch, 2023). We provide evidence that the process underlying both tasks is functionally very similar, and contains about a 78% overlap in in-circuit attention heads. We further present a proof-of-concept intervention experiment, in which we adjust four attention heads in middle layers in order to 'repair' the Colored Objects circuit and make it behave like the IOI circuit. In doing so, we boost accuracy from 49.6% to 93.7% on the Colored Objects task and explain most sources of error. The intervention affects downstream attention heads in specific ways predicted by their interactions in the IOI circuit, indicating that this subcircuit behavior is invariant to the different task inputs. Overall, our results provide evidence that it may yet be possible to explain large language models' behavior in terms of a relatively small number of interpretable task-general algorithmic building blocks and computational components.

{{</citation>}}


### (57/172) A Zero-Shot Language Agent for Computer Control with Structured Reflection (Tao Li et al., 2023)

{{<citation>}}

Tao Li, Gang Li, Zhiwei Deng, Bryan Wang, Yang Li. (2023)  
**A Zero-Shot Language Agent for Computer Control with Structured Reflection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SY, cs.CL, eess-SY  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.08740v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown increasing capacity at planning and executing a high-level goal in a live computer environment (e.g. MiniWoB++). To perform a task, recent works often require a model to learn from trace examples of the task via either supervised learning or few/many-shot prompting. Without these trace examples, it remains a challenge how an agent can autonomously learn and improve its control on a computer, which limits the ability of an agent to perform a new task. We approach this problem with a zero-shot agent that requires no given expert traces. Our agent plans for executable actions on a partially observed environment, and iteratively progresses a task by identifying and learning from its mistakes via self-reflection and structured thought management. On the easy tasks of MiniWoB++, we show that our zero-shot agent often outperforms recent SoTAs, with more efficient reasoning. For tasks with more complexity, our reflective agent performs on par with prior best models, even though previous works had the advantages of accessing expert traces or additional screen information.

{{</citation>}}


### (58/172) Toward Joint Language Modeling for Speech Units and Text (Ju-Chieh Chou et al., 2023)

{{<citation>}}

Ju-Chieh Chou, Chung-Ming Chien, Wei-Ning Hsu, Karen Livescu, Arun Babu, Alexis Conneau, Alexei Baevski, Michael Auli. (2023)  
**Toward Joint Language Modeling for Speech Units and Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08715v1)  

---


**ABSTRACT**  
Speech and text are two major forms of human language. The research community has been focusing on mapping speech to text or vice versa for many years. However, in the field of language modeling, very little effort has been made to model them jointly. In light of this, we explore joint language modeling for speech units and text. Specifically, we compare different speech tokenizers to transform continuous speech signals into discrete units and use different methods to construct mixed speech-text data. We introduce automatic metrics to evaluate how well the joint LM mixes speech and text. We also fine-tune the LM on downstream spoken language understanding (SLU) tasks with different modalities (speech or text) and test its performance to assess the model's learning of shared representations. Our results show that by mixing speech units and text with our proposed mixing techniques, the joint LM improves over a speech-only baseline on SLU tasks and shows zero-shot cross-modal transferability.

{{</citation>}}


### (59/172) Can GPT models be Financial Analysts? An Evaluation of ChatGPT and GPT-4 on mock CFA Exams (Ethan Callanan et al., 2023)

{{<citation>}}

Ethan Callanan, Amarachi Mbakwe, Antony Papadimitriou, Yulong Pei, Mathieu Sibue, Xiaodan Zhu, Zhiqiang Ma, Xiaomo Liu, Sameena Shah. (2023)  
**Can GPT models be Financial Analysts? An Evaluation of ChatGPT and GPT-4 on mock CFA Exams**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL, q-fin-GN  
Keywords: ChatGPT, Few-Shot, Financial, GPT, GPT-4, Language Model, NLP, Natural Language Processing, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.08678v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable performance on a wide range of Natural Language Processing (NLP) tasks, often matching or even beating state-of-the-art task-specific models. This study aims at assessing the financial reasoning capabilities of LLMs. We leverage mock exam questions of the Chartered Financial Analyst (CFA) Program to conduct a comprehensive evaluation of ChatGPT and GPT-4 in financial analysis, considering Zero-Shot (ZS), Chain-of-Thought (CoT), and Few-Shot (FS) scenarios. We present an in-depth analysis of the models' performance and limitations, and estimate whether they would have a chance at passing the CFA exams. Finally, we outline insights into potential strategies and improvements to enhance the applicability of LLMs in finance. In this perspective, we hope this work paves the way for future studies to continue enhancing LLMs for financial reasoning through rigorous evaluation.

{{</citation>}}


### (60/172) LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models (Yixiao Li et al., 2023)

{{<citation>}}

Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao. (2023)  
**LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2310.08659v1)  

---


**ABSTRACT**  
Quantization is an indispensable technique for serving Large Language Models (LLMs) and has recently found its way into LoRA fine-tuning. In this work we focus on the scenario where quantization and LoRA fine-tuning are applied together on a pre-trained model. In such cases it is common to observe a consistent gap in the performance on downstream tasks between full fine-tuning and quantization plus LoRA fine-tuning approach. In response, we propose LoftQ (LoRA-Fine-Tuning-aware Quantization), a novel quantization framework that simultaneously quantizes an LLM and finds a proper low-rank initialization for LoRA fine-tuning. Such an initialization alleviates the discrepancy between the quantized and full-precision model and significantly improves the generalization in downstream tasks. We evaluate our method on natural language understanding, question answering, summarization, and natural language generation tasks. Experiments show that our method is highly effective and outperforms existing quantization methods, especially in the challenging 2-bit and 2/4-bit mixed precision regimes. We will release our code.

{{</citation>}}


### (61/172) Tree-Planner: Efficient Close-loop Task Planning with Large Language Models (Mengkang Hu et al., 2023)

{{<citation>}}

Mengkang Hu, Yao Mu, Xinmiao Yu, Mingyu Ding, Shiguang Wu, Wenqi Shao, Qiguang Chen, Bin Wang, Yu Qiao, Ping Luo. (2023)  
**Tree-Planner: Efficient Close-loop Task Planning with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-RO, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08582v1)  

---


**ABSTRACT**  
This paper studies close-loop task planning, which refers to the process of generating a sequence of skills (a plan) to accomplish a specific goal while adapting the plan based on real-time observations. Recently, prompting Large Language Models (LLMs) to generate actions iteratively has become a prevalent paradigm due to its superior performance and user-friendliness. However, this paradigm is plagued by two inefficiencies: high token consumption and redundant error correction, both of which hinder its scalability for large-scale testing and applications. To address these issues, we propose Tree-Planner, which reframes task planning with LLMs into three distinct phases: plan sampling, action tree construction, and grounded deciding. Tree-Planner starts by using an LLM to sample a set of potential plans before execution, followed by the aggregation of them to form an action tree. Finally, the LLM performs a top-down decision-making process on the tree, taking into account real-time environmental information. Experiments show that Tree-Planner achieves state-of-the-art performance while maintaining high efficiency. By decomposing LLM queries into a single plan-sampling call and multiple grounded-deciding calls, a considerable part of the prompt are less likely to be repeatedly consumed. As a result, token consumption is reduced by 92.2% compared to the previously best-performing model. Additionally, by enabling backtracking on the action tree as needed, the correction process becomes more flexible, leading to a 40.5% decrease in error corrections. Project page: https://tree-planner.github.io/

{{</citation>}}


### (62/172) Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement (Linlu Qiu et al., 2023)

{{<citation>}}

Linlu Qiu, Liwei Jiang, Ximing Lu, Melanie Sclar, Valentina Pyatkin, Chandra Bhagavatula, Bailin Wang, Yoon Kim, Yejin Choi, Nouha Dziri, Xiang Ren. (2023)  
**Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.08559v1)  

---


**ABSTRACT**  
The ability to derive underlying principles from a handful of observations and then generalize to novel situations -- known as inductive reasoning -- is central to human intelligence. Prior work suggests that language models (LMs) often fall short on inductive reasoning, despite achieving impressive success on research benchmarks. In this work, we conduct a systematic study of the inductive reasoning capabilities of LMs through iterative hypothesis refinement, a technique that more closely mirrors the human inductive process than standard input-output prompting. Iterative hypothesis refinement employs a three-step process: proposing, selecting, and refining hypotheses in the form of textual rules. By examining the intermediate rules, we observe that LMs are phenomenal hypothesis proposers (i.e., generating candidate rules), and when coupled with a (task-specific) symbolic interpreter that is able to systematically filter the proposed set of rules, this hybrid approach achieves strong results across inductive reasoning benchmarks that require inducing causal relations, language-like instructions, and symbolic concepts. However, they also behave as puzzling inductive reasoners, showing notable performance gaps in rule induction (i.e., identifying plausible rules) and rule application (i.e., applying proposed rules to instances), suggesting that LMs are proposing hypotheses without being able to actually apply the rules. Through empirical and human analyses, we further reveal several discrepancies between the inductive reasoning processes of LMs and humans, shedding light on both the potentials and limitations of using LMs in inductive reasoning tasks.

{{</citation>}}


### (63/172) Do pretrained Transformers Really Learn In-context by Gradient Descent? (Lingfeng Shen et al., 2023)

{{<citation>}}

Lingfeng Shen, Aayush Mishra, Daniel Khashabi. (2023)  
**Do pretrained Transformers Really Learn In-context by Gradient Descent?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08540v1)  

---


**ABSTRACT**  
Is In-Context Learning (ICL) implicitly equivalent to Gradient Descent (GD)? Several recent works draw analogies between the dynamics of GD and the emergent behavior of ICL in large language models. However, these works make assumptions far from the realistic natural language setting in which language models are trained. Such discrepancies between theory and practice, therefore, necessitate further investigation to validate their applicability.   We start by highlighting the weaknesses in prior works that construct Transformer weights to simulate gradient descent. Their experiments with training Transformers on ICL objective, inconsistencies in the order sensitivity of ICL and GD, sparsity of the constructed weights, and sensitivity to parameter changes are some examples of a mismatch from the real-world setting.   Furthermore, we probe and compare the ICL vs. GD hypothesis in a natural setting. We conduct comprehensive empirical analyses on language models pretrained on natural data (LLaMa-7B). Our comparisons on various performance metrics highlight the inconsistent behavior of ICL and GD as a function of various factors such as datasets, models, and number of demonstrations. We observe that ICL and GD adapt the output distribution of language models differently. These results indicate that the equivalence between ICL and GD is an open hypothesis, requires nuanced considerations and calls for further studies.

{{</citation>}}


### (64/172) LLM-augmented Preference Learning from Natural Language (Inwon Kang et al., 2023)

{{<citation>}}

Inwon Kang, Sikai Ruan, Tyler Ho, Jui-Chien Lin, Farhad Mohsin, Oshani Seneviratne, Lirong Xia. (2023)  
**LLM-augmented Preference Learning from Natural Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08523v1)  

---


**ABSTRACT**  
Finding preferences expressed in natural language is an important but challenging task. State-of-the-art(SotA) methods leverage transformer-based models such as BERT, RoBERTa, etc. and graph neural architectures such as graph attention networks. Since Large Language Models (LLMs) are equipped to deal with larger context lengths and have much larger model sizes than the transformer-based model, we investigate their ability to classify comparative text directly. This work aims to serve as a first step towards using LLMs for the CPC task. We design and conduct a set of experiments that format the classification task into an input prompt for the LLM and a methodology to get a fixed-format response that can be automatically evaluated. Comparing performances with existing methods, we see that pre-trained LLMs are able to outperform the previous SotA models with no fine-tuning involved. Our results show that the LLMs can consistently outperform the SotA when the target text is large -- i.e. composed of multiple sentences --, and are still comparable to the SotA performance in shorter text. We also find that few-shot learning yields better performance than zero-shot learning.

{{</citation>}}


### (65/172) HoneyBee: Progressive Instruction Finetuning of Large Language Models for Materials Science (Yu Song et al., 2023)

{{<citation>}}

Yu Song, Santiago Miret, Huan Zhang, Bang Liu. (2023)  
**HoneyBee: Progressive Instruction Finetuning of Large Language Models for Materials Science**  

---
Primary Category: cs.CL  
Categories: cond-mat-mtrl-sci, cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.08511v1)  

---


**ABSTRACT**  
We propose an instruction-based process for trustworthy data curation in materials science (MatSci-Instruct), which we then apply to finetune a LLaMa-based language model targeted for materials science (HoneyBee). MatSci-Instruct helps alleviate the scarcity of relevant, high-quality materials science textual data available in the open literature, and HoneyBee is the first billion-parameter language model specialized to materials science. In MatSci-Instruct we improve the trustworthiness of generated data by prompting multiple commercially available large language models for generation with an Instructor module (e.g. Chat-GPT) and verification from an independent Verifier module (e.g. Claude). Using MatSci-Instruct, we construct a dataset of multiple tasks and measure the quality of our dataset along multiple dimensions, including accuracy against known facts, relevance to materials science, as well as completeness and reasonableness of the data. Moreover, we iteratively generate more targeted instructions and instruction-data in a finetuning-evaluation-feedback loop leading to progressively better performance for our finetuned HoneyBee models. Our evaluation on the MatSci-NLP benchmark shows HoneyBee's outperformance of existing language models on materials science tasks and iterative improvement in successive stages of instruction-data refinement. We study the quality of HoneyBee's language modeling through automatic evaluation and analyze case studies to further understand the model's capabilities and limitations. Our code and relevant datasets are publicly available at \url{https://github.com/BangLab-UdeM-Mila/NLP4MatSci-HoneyBee}.

{{</citation>}}


### (66/172) The Uncertainty-based Retrieval Framework for Ancient Chinese CWS and POS (Pengyu Wang et al., 2023)

{{<citation>}}

Pengyu Wang, Zhichen Ren. (2023)  
**The Uncertainty-based Retrieval Framework for Ancient Chinese CWS and POS**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.08496v1)  

---


**ABSTRACT**  
Automatic analysis for modern Chinese has greatly improved the accuracy of text mining in related fields, but the study of ancient Chinese is still relatively rare. Ancient text division and lexical annotation are important parts of classical literature comprehension, and previous studies have tried to construct auxiliary dictionary and other fused knowledge to improve the performance. In this paper, we propose a framework for ancient Chinese Word Segmentation and Part-of-Speech Tagging that makes a twofold effort: on the one hand, we try to capture the wordhood semantics; on the other hand, we re-predict the uncertain samples of baseline model by introducing external knowledge. The performance of our architecture outperforms pre-trained BERT with CRF and existing tools such as Jiayan.

{{</citation>}}


### (67/172) Prometheus: Inducing Fine-grained Evaluation Capability in Language Models (Seungone Kim et al., 2023)

{{<citation>}}

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, Minjoon Seo. (2023)  
**Prometheus: Inducing Fine-grained Evaluation Capability in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08491v1)  

---


**ABSTRACT**  
Recently, using a powerful proprietary Large Language Model (LLM) (e.g., GPT-4) as an evaluator for long-form responses has become the de facto standard. However, for practitioners with large-scale evaluation tasks and custom criteria in consideration (e.g., child-readability), using proprietary LLMs as an evaluator is unreliable due to the closed-source nature, uncontrolled versioning, and prohibitive costs. In this work, we propose Prometheus, a fully open-source LLM that is on par with GPT-4's evaluation capabilities when the appropriate reference materials (reference answer, score rubric) are accompanied. We first construct the Feedback Collection, a new dataset that consists of 1K fine-grained score rubrics, 20K instructions, and 100K responses and language feedback generated by GPT-4. Using the Feedback Collection, we train Prometheus, a 13B evaluator LLM that can assess any given long-form text based on customized score rubric provided by the user. Experimental results show that Prometheus scores a Pearson correlation of 0.897 with human evaluators when evaluating with 45 customized score rubrics, which is on par with GPT-4 (0.882), and greatly outperforms ChatGPT (0.392). Furthermore, measuring correlation with GPT-4 with 1222 customized score rubrics across four benchmarks (MT Bench, Vicuna Bench, Feedback Bench, Flask Eval) shows similar trends, bolstering Prometheus's capability as an evaluator LLM. Lastly, Prometheus achieves the highest accuracy on two human preference benchmarks (HHH Alignment & MT Bench Human Judgment) compared to open-sourced reward models explicitly trained on human preference datasets, highlighting its potential as an universal reward model. We open-source our code, dataset, and model at https://github.com/kaistAI/Prometheus.

{{</citation>}}


### (68/172) GraphextQA: A Benchmark for Evaluating Graph-Enhanced Large Language Models (Yuanchun Shen et al., 2023)

{{<citation>}}

Yuanchun Shen, Ruotong Liao, Zhen Han, Yunpu Ma, Volker Tresp. (2023)  
**GraphextQA: A Benchmark for Evaluating Graph-Enhanced Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GNN, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.08487v1)  

---


**ABSTRACT**  
While multi-modal models have successfully integrated information from image, video, and audio modalities, integrating graph modality into large language models (LLMs) remains unexplored. This discrepancy largely stems from the inherent divergence between structured graph data and unstructured text data. Incorporating graph knowledge provides a reliable source of information, enabling potential solutions to address issues in text generation, e.g., hallucination, and lack of domain knowledge. To evaluate the integration of graph knowledge into language models, a dedicated dataset is needed. However, there is currently no benchmark dataset specifically designed for multimodal graph-language models. To address this gap, we propose GraphextQA, a question answering dataset with paired subgraphs, retrieved from Wikidata, to facilitate the evaluation and future development of graph-language models. Additionally, we introduce a baseline model called CrossGNN, which conditions answer generation on the paired graphs by cross-attending question-aware graph features at decoding. The proposed dataset is designed to evaluate graph-language models' ability to understand graphs and make use of it for answer generation. We perform experiments with language-only models and the proposed graph-language model to validate the usefulness of the paired graphs and to demonstrate the difficulty of the task.

{{</citation>}}


### (69/172) Can We Edit Multimodal Large Language Models? (Siyuan Cheng et al., 2023)

{{<citation>}}

Siyuan Cheng, Bozhong Tian, Qingbin Liu, Xi Chen, Yongheng Wang, Huajun Chen, Ningyu Zhang. (2023)  
**Can We Edit Multimodal Large Language Models?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-MM, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.08475v2)  

---


**ABSTRACT**  
In this paper, we focus on editing Multimodal Large Language Models (MLLMs). Compared to editing single-modal LLMs, multimodal model editing is more challenging, which demands a higher level of scrutiny and careful consideration in the editing process. To facilitate research in this area, we construct a new benchmark, dubbed MMEdit, for editing multimodal LLMs and establishing a suite of innovative metrics for evaluation. We conduct comprehensive experiments involving various model editing baselines and analyze the impact of editing different components for multimodal LLMs. Empirically, we notice that previous baselines can implement editing multimodal LLMs to some extent, but the effect is still barely satisfactory, indicating the potential difficulty of this task. We hope that our work can provide the NLP community with insights. Code and dataset are available in https://github.com/zjunlp/EasyEdit.

{{</citation>}}


### (70/172) DistillSpec: Improving Speculative Decoding via Knowledge Distillation (Yongchao Zhou et al., 2023)

{{<citation>}}

Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, Rishabh Agarwal. (2023)  
**DistillSpec: Improving Speculative Decoding via Knowledge Distillation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.08461v1)  

---


**ABSTRACT**  
Speculative decoding (SD) accelerates large language model inference by employing a faster draft model for generating multiple tokens, which are then verified in parallel by the larger target model, resulting in the text generated according to the target model distribution. However, identifying a compact draft model that is well-aligned with the target model is challenging. To tackle this issue, we propose DistillSpec that uses knowledge distillation to better align the draft model with the target model, before applying SD. DistillSpec makes two key design choices, which we demonstrate via systematic study to be crucial to improving the draft and target alignment: utilizing on-policy data generation from the draft model, and tailoring the divergence function to the task and decoding strategy. Notably, DistillSpec yields impressive 10 - 45% speedups over standard SD on a range of standard benchmarks, using both greedy and non-greedy sampling. Furthermore, we combine DistillSpec with lossy SD to achieve fine-grained control over the latency vs. task performance trade-off. Finally, in practical scenarios with models of varying sizes, first using distillation to boost the performance of the target model and then applying DistillSpec to train a well-aligned draft model can reduce decoding latency by 6-10x with minimal performance drop, compared to standard decoding without distillation.

{{</citation>}}


### (71/172) Prompting Large Language Models with Chain-of-Thought for Few-Shot Knowledge Base Question Generation (Yuanyuan Liang et al., 2023)

{{<citation>}}

Yuanyuan Liang, Jianing Wang, Hanlun Zhu, Lei Wang, Weining Qian, Yunshi Lan. (2023)  
**Prompting Large Language Models with Chain-of-Thought for Few-Shot Knowledge Base Question Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, Few-Shot, Language Model, Question Generation  
[Paper Link](http://arxiv.org/abs/2310.08395v1)  

---


**ABSTRACT**  
The task of Question Generation over Knowledge Bases (KBQG) aims to convert a logical form into a natural language question. For the sake of expensive cost of large-scale question annotation, the methods of KBQG under low-resource scenarios urgently need to be developed. However, current methods heavily rely on annotated data for fine-tuning, which is not well-suited for few-shot question generation. The emergence of Large Language Models (LLMs) has shown their impressive generalization ability in few-shot tasks. Inspired by Chain-of-Thought (CoT) prompting, which is an in-context learning strategy for reasoning, we formulate KBQG task as a reasoning problem, where the generation of a complete question is splitted into a series of sub-question generation. Our proposed prompting method KQG-CoT first retrieves supportive logical forms from the unlabeled data pool taking account of the characteristics of the logical form. Then, we write a prompt to explicit the reasoning chain of generating complicated questions based on the selected demonstrations. To further ensure prompt quality, we extend KQG-CoT into KQG-CoT+ via sorting the logical forms by their complexity. We conduct extensive experiments over three public KBQG datasets. The results demonstrate that our prompting method consistently outperforms other prompting baselines on the evaluated datasets. Remarkably, our KQG-CoT+ method could surpass existing few-shot SoTA results of the PathQuestions dataset by 18.25, 10.72, and 10.18 absolute points on BLEU-4, METEOR, and ROUGE-L, respectively.

{{</citation>}}


### (72/172) Towards Better Evaluation of Instruction-Following: A Case-Study in Summarization (Ondrej Skopek et al., 2023)

{{<citation>}}

Ondrej Skopek, Rahul Aralikatte, Sian Gooding, Victor Carbune. (2023)  
**Towards Better Evaluation of Instruction-Following: A Case-Study in Summarization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.08394v1)  

---


**ABSTRACT**  
Despite recent advances, evaluating how well large language models (LLMs) follow user instructions remains an open problem. While evaluation methods of language models have seen a rise in prompt-based approaches, limited work on the correctness of these methods has been conducted. In this work, we perform a meta-evaluation of a variety of metrics to quantify how accurately they measure the instruction-following abilities of LLMs. Our investigation is performed on grounded query-based summarization by collecting a new short-form, real-world dataset riSum, containing $300$ document-instruction pairs with $3$ answers each. All $900$ answers are rated by $3$ human annotators. Using riSum, we analyze agreement between evaluation methods and human judgment. Finally, we propose new LLM-based reference-free evaluation methods that improve upon established baselines and perform on-par with costly reference-based metrics which require high-quality summaries.

{{</citation>}}


### (73/172) Reconstructing Materials Tetrahedron: Challenges in Materials Information Extraction (Kausik Hira et al., 2023)

{{<citation>}}

Kausik Hira, Mohd Zaki, Dhruvil Sheth, Mausam, N M Anoop Krishnan. (2023)  
**Reconstructing Materials Tetrahedron: Challenges in Materials Information Extraction**  

---
Primary Category: cs.CL  
Categories: cond-mat-mtrl-sci, cs-CL, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2310.08383v1)  

---


**ABSTRACT**  
Discovery of new materials has a documented history of propelling human progress for centuries and more. The behaviour of a material is a function of its composition, structure, and properties, which further depend on its processing and testing conditions. Recent developments in deep learning and natural language processing have enabled information extraction at scale from published literature such as peer-reviewed publications, books, and patents. However, this information is spread in multiple formats, such as tables, text, and images, and with little or no uniformity in reporting style giving rise to several machine learning challenges. Here, we discuss, quantify, and document these outstanding challenges in automated information extraction (IE) from materials science literature towards the creation of a large materials science knowledge base. Specifically, we focus on IE from text and tables and outline several challenges with examples. We hope the present work inspires researchers to address the challenges in a coherent fashion, providing to fillip to IE for the materials knowledge base.

{{</citation>}}


### (74/172) Improving Factual Consistency for Knowledge-Grounded Dialogue Systems via Knowledge Enhancement and Alignment (Boyang Xue et al., 2023)

{{<citation>}}

Boyang Xue, Weichao Wang, Hongru Wang, Fei Mi, Rui Wang, Yasheng Wang, Lifeng Shang, Xin Jiang, Qun Liu, Kam-Fai Wong. (2023)  
**Improving Factual Consistency for Knowledge-Grounded Dialogue Systems via Knowledge Enhancement and Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, NLI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08372v1)  

---


**ABSTRACT**  
Pretrained language models (PLMs) based knowledge-grounded dialogue systems are prone to generate responses that are factually inconsistent with the provided knowledge source. In such inconsistent responses, the dialogue models fail to accurately express the external knowledge they rely upon. Inspired by previous work which identified that feed-forward networks (FFNs) within Transformers are responsible for factual knowledge expressions, we investigate two methods to efficiently improve the factual expression capability {of FFNs} by knowledge enhancement and alignment respectively. We first propose \textsc{K-Dial}, which {explicitly} introduces {extended FFNs in Transformers to enhance factual knowledge expressions} given the specific patterns of knowledge-grounded dialogue inputs. Additionally, we apply the reinforcement learning for factual consistency (RLFC) method to implicitly adjust FFNs' expressions in responses by aligning with gold knowledge for the factual consistency preference. To comprehensively assess the factual consistency and dialogue quality of responses, we employ extensive automatic measures and human evaluations including sophisticated fine-grained NLI-based metrics. Experimental results on WoW and CMU\_DoG datasets demonstrate that our methods efficiently enhance the ability of the FFN module to convey factual knowledge, validating the efficacy of improving factual consistency for knowledge-grounded dialogue systems.

{{</citation>}}


### (75/172) From Large Language Models to Knowledge Graphs for Biomarker Discovery in Cancer (Md. Rezaul Karim et al., 2023)

{{<citation>}}

Md. Rezaul Karim, Lina Molinas Comet, Md Shajalal, Oya Beyan, Dietrich Rebholz-Schuhmann, Stefan Decker. (2023)  
**From Large Language Models to Knowledge Graphs for Biomarker Discovery in Cancer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BERT, Knowledge Graph, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.08365v1)  

---


**ABSTRACT**  
Domain experts often rely on up-to-date knowledge for apprehending and disseminating specific biological processes that help them design strategies to develop prevention and therapeutic decision-making. A challenging scenario for artificial intelligence (AI) is using biomedical data (e.g., texts, imaging, omics, and clinical) to provide diagnosis and treatment recommendations for cancerous conditions. Data and knowledge about cancer, drugs, genes, proteins, and their mechanism is spread across structured (knowledge bases (KBs)) and unstructured (e.g., scientific articles) sources. A large-scale knowledge graph (KG) can be constructed by integrating these data, followed by extracting facts about semantically interrelated entities and relations. Such KGs not only allow exploration and question answering (QA) but also allow domain experts to deduce new knowledge. However, exploring and querying large-scale KGs is tedious for non-domain users due to a lack of understanding of the underlying data assets and semantic technologies. In this paper, we develop a domain KG to leverage cancer-specific biomarker discovery and interactive QA. For this, a domain ontology called OncoNet Ontology (ONO) is developed to enable semantic reasoning for validating gene-disease relations. The KG is then enriched by harmonizing the ONO, controlled vocabularies, and additional biomedical concepts from scientific articles by employing BioBERT- and SciBERT-based information extraction (IE) methods. Further, since the biomedical domain is evolving, where new findings often replace old ones, without employing up-to-date findings, there is a high chance an AI system exhibits concept drift while providing diagnosis and treatment. Therefore, we finetuned the KG using large language models (LLMs) based on more recent articles and KBs that might not have been seen by the named entity recognition models.

{{</citation>}}


### (76/172) Not All Demonstration Examples are Equally Beneficial: Reweighting Demonstration Examples for In-Context Learning (Zhe Yang et al., 2023)

{{<citation>}}

Zhe Yang, Damai Dai, Peiyi Wang, Zhifang Sui. (2023)  
**Not All Demonstration Examples are Equally Beneficial: Reweighting Demonstration Examples for In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08309v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently gained the In-Context Learning (ICL) ability with the models scaling up, allowing them to quickly adapt to downstream tasks with only a few demonstration examples prepended in the input sequence. Nonetheless, the current practice of ICL treats all demonstration examples equally, which still warrants improvement, as the quality of examples is usually uneven. In this paper, we investigate how to determine approximately optimal weights for demonstration examples and how to apply them during ICL. To assess the quality of weights in the absence of additional validation data, we design a masked self-prediction (MSP) score that exhibits a strong correlation with the final ICL performance. To expedite the weight-searching process, we discretize the continuous weight space and adopt beam search. With approximately optimal weights obtained, we further propose two strategies to apply them to demonstrations at different model positions. Experimental results on 8 text classification tasks show that our approach outperforms conventional ICL by a large margin. Our code are publicly available at https:github.com/Zhe-Young/WICL.

{{</citation>}}


### (77/172) MProto: Multi-Prototype Network with Denoised Optimal Transport for Distantly Supervised Named Entity Recognition (Shuhui Wu et al., 2023)

{{<citation>}}

Shuhui Wu, Yongliang Shen, Zeqi Tan, Wenqi Ren, Jietian Guo, Shiliang Pu, Weiming Lu. (2023)  
**MProto: Multi-Prototype Network with Denoised Optimal Transport for Distantly Supervised Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.08298v1)  

---


**ABSTRACT**  
Distantly supervised named entity recognition (DS-NER) aims to locate entity mentions and classify their types with only knowledge bases or gazetteers and unlabeled corpus. However, distant annotations are noisy and degrade the performance of NER models. In this paper, we propose a noise-robust prototype network named MProto for the DS-NER task. Different from previous prototype-based NER methods, MProto represents each entity type with multiple prototypes to characterize the intra-class variance among entity representations. To optimize the classifier, each token should be assigned an appropriate ground-truth prototype and we consider such token-prototype assignment as an optimal transport (OT) problem. Furthermore, to mitigate the noise from incomplete labeling, we propose a novel denoised optimal transport (DOT) algorithm. Specifically, we utilize the assignment result between Other class tokens and all prototypes to distinguish unlabeled entity tokens from true negatives. Experiments on several DS-NER benchmarks demonstrate that our MProto achieves state-of-the-art performance. The source code is now available on Github.

{{</citation>}}


### (78/172) Expanding the Vocabulary of BERT for Knowledge Base Construction (Dong Yang et al., 2023)

{{<citation>}}

Dong Yang, Xu Wang, Remzi Celebi. (2023)  
**Expanding the Vocabulary of BERT for Knowledge Base Construction**  

---
Primary Category: cs.CL  
Categories: 68T20, cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.08291v1)  

---


**ABSTRACT**  
Knowledge base construction entails acquiring structured information to create a knowledge base of factual and relational data, facilitating question answering, information retrieval, and semantic understanding. The challenge called "Knowledge Base Construction from Pretrained Language Models" at International Semantic Web Conference 2023 defines tasks focused on constructing knowledge base using language model. Our focus was on Track 1 of the challenge, where the parameters are constrained to a maximum of 1 billion, and the inclusion of entity descriptions within the prompt is prohibited.   Although the masked language model offers sufficient flexibility to extend its vocabulary, it is not inherently designed for multi-token prediction. To address this, we present Vocabulary Expandable BERT for knowledge base construction, which expand the language model's vocabulary while preserving semantic embeddings for newly added words. We adopt task-specific re-pre-training on masked language model to further enhance the language model.   Through experimentation, the results show the effectiveness of our approaches. Our framework achieves F1 score of 0.323 on the hidden test set and 0.362 on the validation set, both data set is provided by the challenge. Notably, our framework adopts a lightweight language model (BERT-base, 0.13 billion parameters) and surpasses the model using prompts directly on large language model (Chatgpt-3, 175 billion parameters). Besides, Token-Recode achieves comparable performances as Re-pretrain. This research advances language understanding models by enabling the direct embedding of multi-token entities, signifying a substantial step forward in link prediction task in knowledge graph and metadata completion in data management.

{{</citation>}}


### (79/172) CP-KGC: Constrained-Prompt Knowledge Graph Completion with Large Language Models (Rui Yang et al., 2023)

{{<citation>}}

Rui Yang, Li Fang, Yi Zhou. (2023)  
**CP-KGC: Constrained-Prompt Knowledge Graph Completion with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08279v1)  

---


**ABSTRACT**  
Knowledge graph completion (KGC) aims to utilize existing knowledge to deduce and infer missing connections within knowledge graphs. Text-based approaches, like SimKGC, have outperformed graph embedding methods, showcasing the promise of inductive KGC. However, the efficacy of text-based methods hinges on the quality of entity textual descriptions. In this paper, we identify the key issue of whether large language models (LLMs) can generate effective text. To mitigate hallucination in LLM-generated text in this paper, we introduce a constraint-based prompt that utilizes the entity and its textual description as contextual constraints to enhance data quality. Our Constrained-Prompt Knowledge Graph Completion (CP-KGC) method demonstrates effective inference under low resource computing conditions and surpasses prior results on the WN18RR and FB15K237 datasets. This showcases the integration of LLMs in KGC tasks and provides new directions for future research.

{{</citation>}}


### (80/172) Impact of Co-occurrence on Factual Knowledge of Large Language Models (Cheongwoong Kang et al., 2023)

{{<citation>}}

Cheongwoong Kang, Jaesik Choi. (2023)  
**Impact of Co-occurrence on Factual Knowledge of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08256v1)  

---


**ABSTRACT**  
Large language models (LLMs) often make factually incorrect responses despite their success in various applications. In this paper, we hypothesize that relying heavily on simple co-occurrence statistics of the pre-training corpora is one of the main factors that cause factual errors. Our results reveal that LLMs are vulnerable to the co-occurrence bias, defined as preferring frequently co-occurred words over the correct answer. Consequently, LLMs struggle to recall facts whose subject and object rarely co-occur in the pre-training dataset although they are seen during finetuning. We show that co-occurrence bias remains despite scaling up model sizes or finetuning. Therefore, we suggest finetuning on a debiased dataset to mitigate the bias by filtering out biased samples whose subject-object co-occurrence count is high. Although debiased finetuning allows LLMs to memorize rare facts in the training set, it is not effective in recalling rare facts unseen during finetuning. Further research in mitigation will help build reliable language models by preventing potential errors. The code is available at \url{https://github.com/CheongWoong/impact_of_cooccurrence}.

{{</citation>}}


### (81/172) Who Said That? Benchmarking Social Media AI Detection (Wanyun Cui et al., 2023)

{{<citation>}}

Wanyun Cui, Linqiu Zhang, Qianle Wang, Shuyang Cai. (2023)  
**Who Said That? Benchmarking Social Media AI Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Social Media  
[Paper Link](http://arxiv.org/abs/2310.08240v1)  

---


**ABSTRACT**  
AI-generated text has proliferated across various online platforms, offering both transformative prospects and posing significant risks related to misinformation and manipulation. Addressing these challenges, this paper introduces SAID (Social media AI Detection), a novel benchmark developed to assess AI-text detection models' capabilities in real social media platforms. It incorporates real AI-generate text from popular social media platforms like Zhihu and Quora. Unlike existing benchmarks, SAID deals with content that reflects the sophisticated strategies employed by real AI users on the Internet which may evade detection or gain visibility, providing a more realistic and challenging evaluation landscape. A notable finding of our study, based on the Zhihu dataset, reveals that annotators can distinguish between AI-generated and human-generated texts with an average accuracy rate of 96.5%. This finding necessitates a re-evaluation of human capability in recognizing AI-generated text in today's widely AI-influenced environment. Furthermore, we present a new user-oriented AI-text detection challenge focusing on the practicality and effectiveness of identifying AI-generated text based on user information and multiple responses. The experimental results demonstrate that conducting detection tasks on actual social media platforms proves to be more challenging compared to traditional simulated AI-text detection, resulting in a decreased accuracy. On the other hand, user-oriented AI-generated text detection significantly improve the accuracy of detection.

{{</citation>}}


### (82/172) Language Models are Universal Embedders (Xin Zhang et al., 2023)

{{<citation>}}

Xin Zhang, Zehan Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, Min Zhang. (2023)  
**Language Models are Universal Embedders**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08232v1)  

---


**ABSTRACT**  
In the large language model (LLM) revolution, embedding is a key component of various systems. For example, it is used to retrieve knowledge or memories for LLMs, to build content moderation filters, etc. As such cases span from English to other natural or programming languages, from retrieval to classification and beyond, it is desirable to build a unified embedding model rather than dedicated ones for each scenario. In this work, we make an initial step towards this goal, demonstrating that multiple languages (both natural and programming) pre-trained transformer decoders can embed universally when finetuned on limited English data. We provide a comprehensive practice with thorough evaluations. On English MTEB, our models achieve competitive performance on different embedding tasks by minimal training data. On other benchmarks, such as multilingual classification and code search, our models (without any supervision) perform comparably to, or even surpass heavily supervised baselines and/or APIs. These results provide evidence of a promising path towards building powerful unified embedders that can be applied across tasks and languages.

{{</citation>}}


### (83/172) SimCKP: Simple Contrastive Learning of Keyphrase Representations (Minseok Choi et al., 2023)

{{<citation>}}

Minseok Choi, Chaeheon Gwak, Seho Kim, Si Hyeong Kim, Jaegul Choo. (2023)  
**SimCKP: Simple Contrastive Learning of Keyphrase Representations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.08221v1)  

---


**ABSTRACT**  
Keyphrase generation (KG) aims to generate a set of summarizing words or phrases given a source document, while keyphrase extraction (KE) aims to identify them from the text. Because the search space is much smaller in KE, it is often combined with KG to predict keyphrases that may or may not exist in the corresponding document. However, current unified approaches adopt sequence labeling and maximization-based generation that primarily operate at a token level, falling short in observing and scoring keyphrases as a whole. In this work, we propose SimCKP, a simple contrastive learning framework that consists of two stages: 1) An extractor-generator that extracts keyphrases by learning context-aware phrase-level representations in a contrastive manner while also generating keyphrases that do not appear in the document; 2) A reranker that adapts scores for each generated phrase by likewise aligning their representations with the corresponding document. Experimental results on multiple benchmark datasets demonstrate the effectiveness of our proposed approach, which outperforms the state-of-the-art models by a significant margin.

{{</citation>}}


### (84/172) Visual Question Generation in Bengali (Mahmud Hasan et al., 2023)

{{<citation>}}

Mahmud Hasan, Labiba Islam, Jannatul Ferdous Ruma, Tasmiah Tahsin Mayeesha, Rashedur M. Rahman. (2023)  
**Visual Question Generation in Bengali**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, QA, Question Generation  
[Paper Link](http://arxiv.org/abs/2310.08187v1)  

---


**ABSTRACT**  
The task of Visual Question Generation (VQG) is to generate human-like questions relevant to the given image. As VQG is an emerging research field, existing works tend to focus only on resource-rich language such as English due to the availability of datasets. In this paper, we propose the first Bengali Visual Question Generation task and develop a novel transformer-based encoder-decoder architecture that generates questions in Bengali when given an image. We propose multiple variants of models - (i) image-only: baseline model of generating questions from images without additional information, (ii) image-category and image-answer-category: guided VQG where we condition the model to generate questions based on the answer and the category of expected question. These models are trained and evaluated on the translated VQAv2.0 dataset. Our quantitative and qualitative results establish the first state of the art models for VQG task in Bengali and demonstrate that our models are capable of generating grammatically correct and relevant questions. Our quantitative results show that our image-cat model achieves a BLUE-1 score of 33.12 and BLEU-3 score of 7.56 which is the highest of the other two variants. We also perform a human evaluation to assess the quality of the generation tasks. Human evaluation suggests that image-cat model is capable of generating goal-driven and attribute-specific questions and also stays relevant to the corresponding image.

{{</citation>}}


### (85/172) EIPE-text: Evaluation-Guided Iterative Plan Extraction for Long-Form Narrative Text Generation (Wang You et al., 2023)

{{<citation>}}

Wang You, Wenshan Wu, Yaobo Liang, Shaoguang Mao, Chenfei Wu, Maosong Cao, Yuzhe Cai, Yiduo Guo, Yan Xia, Furu Wei, Nan Duan. (2023)  
**EIPE-text: Evaluation-Guided Iterative Plan Extraction for Long-Form Narrative Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, QA, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.08185v1)  

---


**ABSTRACT**  
Plan-and-Write is a common hierarchical approach in long-form narrative text generation, which first creates a plan to guide the narrative writing. Following this approach, several studies rely on simply prompting large language models for planning, which often yields suboptimal results. In this paper, we propose a new framework called Evaluation-guided Iterative Plan Extraction for long-form narrative text generation (EIPE-text), which extracts plans from the corpus of narratives and utilizes the extracted plans to construct a better planner. EIPE-text has three stages: plan extraction, learning, and inference. In the plan extraction stage, it iteratively extracts and improves plans from the narrative corpus and constructs a plan corpus. We propose a question answer (QA) based evaluation mechanism to automatically evaluate the plans and generate detailed plan refinement instructions to guide the iterative improvement. In the learning stage, we build a better planner by fine-tuning with the plan corpus or in-context learning with examples in the plan corpus. Finally, we leverage a hierarchical approach to generate long-form narratives. We evaluate the effectiveness of EIPE-text in the domains of novels and storytelling. Both GPT-4-based evaluations and human evaluations demonstrate that our method can generate more coherent and relevant long-form narratives. Our code will be released in the future.

{{</citation>}}


### (86/172) Exploring the Cognitive Knowledge Structure of Large Language Models: An Educational Diagnostic Assessment Approach (Zheyuan Zhang et al., 2023)

{{<citation>}}

Zheyuan Zhang, Jifan Yu, Juanzi Li, Lei Hou. (2023)  
**Exploring the Cognitive Knowledge Structure of Large Language Models: An Educational Diagnostic Assessment Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08172v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have not only exhibited exceptional performance across various tasks, but also demonstrated sparks of intelligence. Recent studies have focused on assessing their capabilities on human exams and revealed their impressive competence in different domains. However, cognitive research on the overall knowledge structure of LLMs is still lacking. In this paper, based on educational diagnostic assessment method, we conduct an evaluation using MoocRadar, a meticulously annotated human test dataset based on Bloom Taxonomy. We aim to reveal the knowledge structures of LLMs and gain insights of their cognitive capabilities. This research emphasizes the significance of investigating LLMs' knowledge and understanding the disparate cognitive patterns of LLMs. By shedding light on models' knowledge, researchers can advance development and utilization of LLMs in a more informed and effective manner.

{{</citation>}}


### (87/172) Multiclass Classification of Policy Documents with Large Language Models (Erkan Gunes et al., 2023)

{{<citation>}}

Erkan Gunes, Christoffer Koch Florczak. (2023)  
**Multiclass Classification of Policy Documents with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08167v1)  

---


**ABSTRACT**  
Classifying policy documents into policy issue topics has been a long-time effort in political science and communication disciplines. Efforts to automate text classification processes for social science research purposes have so far achieved remarkable results, but there is still a large room for progress. In this work, we test the prediction performance of an alternative strategy, which requires human involvement much less than full manual coding. We use the GPT 3.5 and GPT 4 models of the OpenAI, which are pre-trained instruction-tuned Large Language Models (LLM), to classify congressional bills and congressional hearings into Comparative Agendas Project's 21 major policy issue topics. We propose three use-case scenarios and estimate overall accuracies ranging from %58-83 depending on scenario and GPT model employed. The three scenarios aims at minimal, moderate, and major human interference, respectively. Overall, our results point towards the insufficiency of complete reliance on GPT with minimal human intervention, an increasing accuracy along with the human effort exerted, and a surprisingly high accuracy achieved in the most humanly demanding use-case. However, the superior use-case achieved the %83 accuracy on the %65 of the data in which the two models agreed, suggesting that a similar approach to ours can be relatively easily implemented and allow for mostly automated coding of a majority of a given dataset. This could free up resources allowing manual human coding of the remaining %35 of the data to achieve an overall higher level of accuracy while reducing costs significantly.

{{</citation>}}


### (88/172) Ziya-VL: Bilingual Large Vision-Language Model via Multi-Task Instruction Tuning (Junyu Lu et al., 2023)

{{<citation>}}

Junyu Lu, Dixiang Zhang, Xiaojun Wu, Xinyu Gao, Ruyi Gan, Jiaxing Zhang, Yan Song, Pingjian Zhang. (2023)  
**Ziya-VL: Bilingual Large Vision-Language Model via Multi-Task Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08166v1)  

---


**ABSTRACT**  
Recent advancements enlarge the capabilities of large language models (LLMs) in zero-shot image-to-text generation and understanding by integrating multi-modal inputs. However, such success is typically limited to English scenarios due to the lack of large-scale and high-quality non-English multi-modal resources, making it extremely difficult to establish competitive counterparts in other languages. In this paper, we introduce the Ziya-VL series, a set of bilingual large-scale vision-language models (LVLMs) designed to incorporate visual semantics into LLM for multi-modal dialogue. Composed of Ziya-VL-Base and Ziya-VL-Chat, our models adopt the Querying Transformer from BLIP-2, further exploring the assistance of optimization schemes such as instruction tuning, multi-stage training and low-rank adaptation module for visual-language alignment. In addition, we stimulate the understanding ability of GPT-4 in multi-modal scenarios, translating our gathered English image-text datasets into Chinese and generating instruction-response through the in-context learning method. The experiment results demonstrate that compared to the existing LVLMs, Ziya-VL achieves competitive performance across a wide range of English-only tasks including zero-shot image-text retrieval, image captioning, and visual question answering. The evaluation leaderboard accessed by GPT-4 also indicates that our models possess satisfactory image-text understanding and generation capabilities in Chinese multi-modal scenario dialogues. Code, demo and models are available at ~\url{https://huggingface.co/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1}.

{{</citation>}}


### (89/172) Context Compression for Auto-regressive Transformers with Sentinel Tokens (Siyu Ren et al., 2023)

{{<citation>}}

Siyu Ren, Qi Jia, Kenny Q. Zhu. (2023)  
**Context Compression for Auto-regressive Transformers with Sentinel Tokens**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08152v1)  

---


**ABSTRACT**  
The quadratic complexity of the attention module makes it gradually become the bulk of compute in Transformer-based LLMs during generation. Moreover, the excessive key-value cache that arises when dealing with long inputs also brings severe issues on memory footprint and inference latency. In this work, we propose a plug-and-play approach that is able to incrementally compress the intermediate activation of a specified span of tokens into compact ones, thereby reducing both memory and computational cost when processing subsequent context. Experiments on both in-domain language modeling and zero-shot open-ended document generation demonstrate the advantage of our approach over sparse attention baselines in terms of fluency, n-gram matching, and semantic similarity. At last, we comprehensively profile the benefit of context compression on improving the system throughout. Code is available at https://github.com/DRSY/KV_Compression.

{{</citation>}}


### (90/172) On the Relevance of Phoneme Duration Variability of Synthesized Training Data for Automatic Speech Recognition (Nick Rossenbach et al., 2023)

{{<citation>}}

Nick Rossenbach, Benedikt Hilmes, Ralf Schlüter. (2023)  
**On the Relevance of Phoneme Duration Variability of Synthesized Training Data for Automatic Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.08132v1)  

---


**ABSTRACT**  
Synthetic data generated by text-to-speech (TTS) systems can be used to improve automatic speech recognition (ASR) systems in low-resource or domain mismatch tasks. It has been shown that TTS-generated outputs still do not have the same qualities as real data. In this work we focus on the temporal structure of synthetic data and its relation to ASR training. By using a novel oracle setup we show how much the degradation of synthetic data quality is influenced by duration modeling in non-autoregressive (NAR) TTS. To get reference phoneme durations we use two common alignment methods, a hidden Markov Gaussian-mixture model (HMM-GMM) aligner and a neural connectionist temporal classification (CTC) aligner. Using a simple algorithm based on random walks we shift phoneme duration distributions of the TTS system closer to real durations, resulting in an improvement of an ASR system using synthetic data in a semi-supervised setting.

{{</citation>}}


### (91/172) Who Wrote it and Why? Prompting Large-Language Models for Authorship Verification (Chia-Yu Hung et al., 2023)

{{<citation>}}

Chia-Yu Hung, Zhiqiang Hu, Yujia Hu, Roy Ka-Wei Lee. (2023)  
**Who Wrote it and Why? Prompting Large-Language Models for Authorship Verification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.08123v1)  

---


**ABSTRACT**  
Authorship verification (AV) is a fundamental task in natural language processing (NLP) and computational linguistics, with applications in forensic analysis, plagiarism detection, and identification of deceptive content. Existing AV techniques, including traditional stylometric and deep learning approaches, face limitations in terms of data requirements and lack of explainability. To address these limitations, this paper proposes PromptAV, a novel technique that leverages Large-Language Models (LLMs) for AV by providing step-by-step stylometric explanation prompts. PromptAV outperforms state-of-the-art baselines, operates effectively with limited training data, and enhances interpretability through intuitive explanations, showcasing its potential as an effective and interpretable solution for the AV task.

{{</citation>}}


### (92/172) QASiNa: Religious Domain Question Answering using Sirah Nabawiyah (Muhammad Razif Rizqullah et al., 2023)

{{<citation>}}

Muhammad Razif Rizqullah, Ayu Purwarianti, Alham Fikri Aji. (2023)  
**QASiNa: Religious Domain Question Answering using Sirah Nabawiyah**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, GPT-4, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.08102v1)  

---


**ABSTRACT**  
Nowadays, Question Answering (QA) tasks receive significant research focus, particularly with the development of Large Language Model (LLM) such as Chat GPT [1]. LLM can be applied to various domains, but it contradicts the principles of information transmission when applied to the Islamic domain. In Islam we strictly regulates the sources of information and who can give interpretations or tafseer for that sources [2]. The approach used by LLM to generate answers based on its own interpretation is similar to the concept of tafseer, LLM is neither an Islamic expert nor a human which is not permitted in Islam. Indonesia is the country with the largest Islamic believer population in the world [3]. With the high influence of LLM, we need to make evaluation of LLM in religious domain. Currently, there is only few religious QA dataset available and none of them using Sirah Nabawiyah especially in Indonesian Language. In this paper, we propose the Question Answering Sirah Nabawiyah (QASiNa) dataset, a novel dataset compiled from Sirah Nabawiyah literatures in Indonesian language. We demonstrate our dataset by using mBERT [4], XLM-R [5], and IndoBERT [6] which fine-tuned with Indonesian translation of SQuAD v2.0 [7]. XLM-R model returned the best performance on QASiNa with EM of 61.20, F1-Score of 75.94, and Substring Match of 70.00. We compare XLM-R performance with Chat GPT-3.5 and GPT-4 [1]. Both Chat GPT version returned lower EM and F1-Score with higher Substring Match, the gap of EM and Substring Match get wider in GPT-4. The experiment indicate that Chat GPT tends to give excessive interpretations as evidenced by its higher Substring Match scores compared to EM and F1-Score, even after providing instruction and context. This concludes Chat GPT is unsuitable for question answering task in religious domain especially for Islamic religion.

{{</citation>}}


### (93/172) Promptor: A Conversational and Autonomous Prompt Generation Agent for Intelligent Text Entry Techniques (Junxiao Shen et al., 2023)

{{<citation>}}

Junxiao Shen, John J. Dudley, Jingyao Zheng, Bill Byrne, Per Ola Kristensson. (2023)  
**Promptor: A Conversational and Autonomous Prompt Generation Agent for Intelligent Text Entry Techniques**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.08101v1)  

---


**ABSTRACT**  
Text entry is an essential task in our day-to-day digital interactions. Numerous intelligent features have been developed to streamline this process, making text entry more effective, efficient, and fluid. These improvements include sentence prediction and user personalization. However, as deep learning-based language models become the norm for these advanced features, the necessity for data collection and model fine-tuning increases. These challenges can be mitigated by harnessing the in-context learning capability of large language models such as GPT-3.5. This unique feature allows the language model to acquire new skills through prompts, eliminating the need for data collection and fine-tuning. Consequently, large language models can learn various text prediction techniques. We initially showed that, for a sentence prediction task, merely prompting GPT-3.5 surpassed a GPT-2 backed system and is comparable with a fine-tuned GPT-3.5 model, with the latter two methods requiring costly data collection, fine-tuning and post-processing. However, the task of prompting large language models to specialize in specific text prediction tasks can be challenging, particularly for designers without expertise in prompt engineering. To address this, we introduce Promptor, a conversational prompt generation agent designed to engage proactively with designers. Promptor can automatically generate complex prompts tailored to meet specific needs, thus offering a solution to this challenge. We conducted a user study involving 24 participants creating prompts for three intelligent text entry tasks, half of the participants used Promptor while the other half designed prompts themselves. The results show that Promptor-designed prompts result in a 35% increase in similarity and 22% in coherence over those by designers.

{{</citation>}}


### (94/172) ClimateNLP: Analyzing Public Sentiment Towards Climate Change Using Natural Language Processing (Ajay Krishnan T. K. et al., 2023)

{{<citation>}}

Ajay Krishnan T. K., V. S. Anoop. (2023)  
**ClimateNLP: Analyzing Public Sentiment Towards Climate Change Using Natural Language Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP, Natural Language Processing, Twitter  
[Paper Link](http://arxiv.org/abs/2310.08099v1)  

---


**ABSTRACT**  
Climate change's impact on human health poses unprecedented and diverse challenges. Unless proactive measures based on solid evidence are implemented, these threats will likely escalate and continue to endanger human well-being. The escalating advancements in information and communication technologies have facilitated the widespread availability and utilization of social media platforms. Individuals utilize platforms such as Twitter and Facebook to express their opinions, thoughts, and critiques on diverse subjects, encompassing the pressing issue of climate change. The proliferation of climate change-related content on social media necessitates comprehensive analysis to glean meaningful insights. This paper employs natural language processing (NLP) techniques to analyze climate change discourse and quantify the sentiment of climate change-related tweets. We use ClimateBERT, a pretrained model fine-tuned specifically for the climate change domain. The objective is to discern the sentiment individuals express and uncover patterns in public opinion concerning climate change. Analyzing tweet sentiments allows a deeper comprehension of public perceptions, concerns, and emotions about this critical global challenge. The findings from this experiment unearth valuable insights into public sentiment and the entities associated with climate change discourse. Policymakers, researchers, and organizations can leverage such analyses to understand public perceptions, identify influential actors, and devise informed strategies to address climate change challenges.

{{</citation>}}


### (95/172) Low-Resource Clickbait Spoiling for Indonesian via Question Answering (Ni Putu Intan Maharani et al., 2023)

{{<citation>}}

Ni Putu Intan Maharani, Ayu Purwarianti, Alham Fikri Aji. (2023)  
**Low-Resource Clickbait Spoiling for Indonesian via Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Low-Resource, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.08085v1)  

---


**ABSTRACT**  
Clickbait spoiling aims to generate a short text to satisfy the curiosity induced by a clickbait post. As it is a newly introduced task, the dataset is only available in English so far. Our contributions include the construction of manually labeled clickbait spoiling corpus in Indonesian and an evaluation on using cross-lingual zero-shot question answering-based models to tackle clikcbait spoiling for low-resource language like Indonesian. We utilize selection of multilingual language models. The experimental results suggest that XLM-RoBERTa (large) model outperforms other models for phrase and passage spoilers, meanwhile, mDeBERTa (base) model outperforms other models for multipart spoilers.

{{</citation>}}


### (96/172) To token or not to token: A Comparative Study of Text Representations for Cross-Lingual Transfer (Md Mushfiqur Rahman et al., 2023)

{{<citation>}}

Md Mushfiqur Rahman, Fardin Ahsan Sakib, Fahim Faisal, Antonios Anastasopoulos. (2023)  
**To token or not to token: A Comparative Study of Text Representations for Cross-Lingual Transfer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, NER  
[Paper Link](http://arxiv.org/abs/2310.08078v1)  

---


**ABSTRACT**  
Choosing an appropriate tokenization scheme is often a bottleneck in low-resource cross-lingual transfer. To understand the downstream implications of text representation choices, we perform a comparative analysis on language models having diverse text representation modalities including 2 segmentation-based models (\texttt{BERT}, \texttt{mBERT}), 1 image-based model (\texttt{PIXEL}), and 1 character-level model (\texttt{CANINE}). First, we propose a scoring Language Quotient (LQ) metric capable of providing a weighted representation of both zero-shot and few-shot evaluation combined. Utilizing this metric, we perform experiments comprising 19 source languages and 133 target languages on three tasks (POS tagging, Dependency parsing, and NER). Our analysis reveals that image-based models excel in cross-lingual transfer when languages are closely related and share visually similar scripts. However, for tasks biased toward word meaning (POS, NER), segmentation-based models prove to be superior. Furthermore, in dependency parsing tasks where word relationships play a crucial role, models with their character-level focus, outperform others. Finally, we propose a recommendation scheme based on our findings to guide model selection according to task and language requirements.

{{</citation>}}


### (97/172) Training Generative Question-Answering on Synthetic Data Obtained from an Instruct-tuned Model (Kosuke Takahashi et al., 2023)

{{<citation>}}

Kosuke Takahashi, Takahiro Omi, Kosuke Arima, Tatsuya Ishigaki. (2023)  
**Training Generative Question-Answering on Synthetic Data Obtained from an Instruct-tuned Model**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-CL, cs.CL  
Keywords: GPT, QA  
[Paper Link](http://arxiv.org/abs/2310.08072v2)  

---


**ABSTRACT**  
This paper presents a simple and cost-effective method for synthesizing data to train question-answering systems. For training, fine-tuning GPT models is a common practice in resource-rich languages like English, however, it becomes challenging for non-English languages due to the scarcity of sufficient question-answer (QA) pairs. Existing approaches use question and answer generators trained on human-authored QA pairs, which involves substantial human expenses. In contrast, we use an instruct-tuned model to generate QA pairs in a zero-shot or few-shot manner. We conduct experiments to compare various strategies for obtaining QA pairs from the instruct-tuned model. The results demonstrate that a model trained on our proposed synthetic data achieves comparable performance to a model trained on manually curated datasets, without incurring human costs.

{{</citation>}}


### (98/172) QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models (Jing Liu et al., 2023)

{{<citation>}}

Jing Liu, Ruihao Gong, Xiuying Wei, Zhiwei Dong, Jianfei Cai, Bohan Zhuang. (2023)  
**QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model, NLP, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2310.08041v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) excel in NLP, but their demands hinder their widespread deployment. While Quantization-Aware Training (QAT) offers a solution, its extensive training costs make Post-Training Quantization (PTQ) a more practical approach for LLMs. In existing studies, activation outliers in particular channels are identified as the bottleneck to PTQ accuracy. They propose to transform the magnitudes from activations to weights, which however offers limited alleviation or suffers from unstable gradients, resulting in a severe performance drop at low-bitwidth. In this paper, we propose QLLM, an accurate and efficient low-bitwidth PTQ method designed for LLMs. QLLM introduces an adaptive channel reassembly technique that reallocates the magnitude of outliers to other channels, thereby mitigating their impact on the quantization range. This is achieved by channel disassembly and channel assembly, which first breaks down the outlier channels into several sub-channels to ensure a more balanced distribution of activation magnitudes. Then similar channels are merged to maintain the original channel number for efficiency. Additionally, an adaptive strategy is designed to autonomously determine the optimal number of sub-channels for channel disassembly. To further compensate for the performance loss caused by quantization, we propose an efficient tuning method that only learns a small number of low-rank weights while freezing the pre-trained quantized model. After training, these low-rank parameters can be fused into the frozen weights without affecting inference. Extensive experiments on LLaMA-1 and LLaMA-2 show that QLLM can obtain accurate quantized models efficiently. For example, QLLM quantizes the 4-bit LLaMA-2-70B within 10 hours on a single A100-80G GPU, outperforming the previous state-of-the-art method by 7.89% on the average accuracy across five zero-shot tasks.

{{</citation>}}


### (99/172) Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection (Yi Dai et al., 2023)

{{<citation>}}

Yi Dai, Hao Lang, Kaisheng Zeng, Fei Huang, Yongbin Li. (2023)  
**Exploring Large Language Models for Multi-Modal Out-of-Distribution Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08027v1)  

---


**ABSTRACT**  
Out-of-distribution (OOD) detection is essential for reliable and trustworthy machine learning. Recent multi-modal OOD detection leverages textual information from in-distribution (ID) class names for visual OOD detection, yet it currently neglects the rich contextual information of ID classes. Large language models (LLMs) encode a wealth of world knowledge and can be prompted to generate descriptive features for each class. Indiscriminately using such knowledge causes catastrophic damage to OOD detection due to LLMs' hallucinations, as is observed by our analysis. In this paper, we propose to apply world knowledge to enhance OOD detection performance through selective generation from LLMs. Specifically, we introduce a consistency-based uncertainty calibration method to estimate the confidence score of each generation. We further extract visual objects from each image to fully capitalize on the aforementioned world knowledge. Extensive experiments demonstrate that our method consistently outperforms the state-of-the-art.

{{</citation>}}


### (100/172) Harnessing Large Language Models' Empathetic Response Generation Capabilities for Online Mental Health Counselling Support (Siyuan Brandon Loh et al., 2023)

{{<citation>}}

Siyuan Brandon Loh, Aravind Sesagiri Raamkumar. (2023)  
**Harnessing Large Language Models' Empathetic Response Generation Capabilities for Online Mental Health Counselling Support**  

---
Primary Category: cs.CL  
Categories: I-2, cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, Falcon, GPT, Language Model, PaLM, T5  
[Paper Link](http://arxiv.org/abs/2310.08017v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable performance across various information-seeking and reasoning tasks. These computational systems drive state-of-the-art dialogue systems, such as ChatGPT and Bard. They also carry substantial promise in meeting the growing demands of mental health care, albeit relatively unexplored. As such, this study sought to examine LLMs' capability to generate empathetic responses in conversations that emulate those in a mental health counselling setting. We selected five LLMs: version 3.5 and version 4 of the Generative Pre-training (GPT), Vicuna FastChat-T5, Pathways Language Model (PaLM) version 2, and Falcon-7B-Instruct. Based on a simple instructional prompt, these models responded to utterances derived from the EmpatheticDialogues (ED) dataset. Using three empathy-related metrics, we compared their responses to those from traditional response generation dialogue systems, which were fine-tuned on the ED dataset, along with human-generated responses. Notably, we discovered that responses from the LLMs were remarkably more empathetic in most scenarios. We position our findings in light of catapulting advancements in creating empathetic conversational systems.

{{</citation>}}


### (101/172) Clustering of Spell Variations for Proper Nouns Transliterated from the other languages (Prathamesh Pawar, 2023)

{{<citation>}}

Prathamesh Pawar. (2023)  
**Clustering of Spell Variations for Proper Nouns Transliterated from the other languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.07962v1)  

---


**ABSTRACT**  
One of the prominent problems with processing and operating on text data is the non uniformity of it. Due to the change in the dialects and languages, the caliber of translation is low. This creates a unique problem while using NLP in text data; which is the spell variation arising from the inconsistent translations and transliterations. This problem can also be further aggravated by the human error arising from the various ways to write a Proper Noun from an Indian language into its English equivalent. Translating proper nouns originating from Indian languages can be complicated as some proper nouns are also used as common nouns which might be taken literally. Applications of NLP that require addresses, names and other proper nouns face this problem frequently. We propose a method to cluster these spell variations for proper nouns using ML techniques and mathematical similarity equations. We aimed to use Affinity Propagation to determine relative similarity between the tokens. The results are augmented by filtering the token-variation pair by a similarity threshold. We were able to reduce the spell variations by a considerable amount. This application can significantly reduce the amount of human annotation efforts needed for data cleansing and formatting.

{{</citation>}}


## cs.LG (36)



### (102/172) Question Answering for Electronic Health Records: A Scoping Review of datasets and models (Jayetri Bardhan et al., 2023)

{{<citation>}}

Jayetri Bardhan, Kirk Roberts, Daisy Zhe Wang. (2023)  
**Question Answering for Electronic Health Records: A Scoping Review of datasets and models**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Google, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.08759v1)  

---


**ABSTRACT**  
Question Answering (QA) systems on patient-related data can assist both clinicians and patients. They can, for example, assist clinicians in decision-making and enable patients to have a better understanding of their medical history. Significant amounts of patient data are stored in Electronic Health Records (EHRs), making EHR QA an important research area. In EHR QA, the answer is obtained from the medical record of the patient. Because of the differences in data format and modality, this differs greatly from other medical QA tasks that employ medical websites or scientific papers to retrieve answers, making it critical to research EHR question answering. This study aimed to provide a methodological review of existing works on QA over EHRs. We searched for articles from January 1st, 2005 to September 30th, 2023 in four digital sources including Google Scholar, ACL Anthology, ACM Digital Library, and PubMed to collect relevant publications on EHR QA. 4111 papers were identified for our study, and after screening based on our inclusion criteria, we obtained a total of 47 papers for further study. Out of the 47 papers, 25 papers were about EHR QA datasets, and 37 papers were about EHR QA models. It was observed that QA on EHRs is relatively new and unexplored. Most of the works are fairly recent. Also, it was observed that emrQA is by far the most popular EHR QA dataset, both in terms of citations and usage in other papers. Furthermore, we identified the different models used in EHR QA along with the evaluation metrics used for these models.

{{</citation>}}


### (103/172) Detection and prediction of clopidogrel treatment failures using longitudinal structured electronic health records (Samuel Kim et al., 2023)

{{<citation>}}

Samuel Kim, In Gu Sean Lee, Mijeong Irene Ban, Jane Chiang. (2023)  
**Detection and prediction of clopidogrel treatment failures using longitudinal structured electronic health records**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08757v1)  

---


**ABSTRACT**  
We propose machine learning algorithms to automatically detect and predict clopidogrel treatment failure using longitudinal structured electronic health records (EHR). By drawing analogies between natural language and structured EHR, we introduce various machine learning algorithms used in natural language processing (NLP) applications to build models for treatment failure detection and prediction. In this regard, we generated a cohort of patients with clopidogrel prescriptions from UK Biobank and annotated if the patients had treatment failure events within one year of the first clopidogrel prescription; out of 502,527 patients, 1,824 patients were identified as treatment failure cases, and 6,859 patients were considered as control cases. From the dataset, we gathered diagnoses, prescriptions, and procedure records together per patient and organized them into visits with the same date to build models. The models were built for two different tasks, i.e., detection and prediction, and the experimental results showed that time series models outperform bag-of-words approaches in both tasks. In particular, a Transformer-based model, namely BERT, could reach 0.928 AUC in detection tasks and 0.729 AUC in prediction tasks. BERT also showed competence over other time series models when there is not enough training data, because it leverages the pre-training procedure using large unlabeled data.

{{</citation>}}


### (104/172) Constrained Bayesian Optimization with Adaptive Active Learning of Unknown Constraints (Fengxue Zhang et al., 2023)

{{<citation>}}

Fengxue Zhang, Zejie Zhu, Yuxin Chen. (2023)  
**Constrained Bayesian Optimization with Adaptive Active Learning of Unknown Constraints**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.08751v1)  

---


**ABSTRACT**  
Optimizing objectives under constraints, where both the objectives and constraints are black box functions, is a common scenario in real-world applications such as scientific experimental design, design of medical therapies, and industrial process optimization. One popular approach to handling these complex scenarios is Bayesian Optimization (BO). In terms of theoretical behavior, BO is relatively well understood in the unconstrained setting, where its principles have been well explored and validated. However, when it comes to constrained Bayesian optimization (CBO), the existing framework often relies on heuristics or approximations without the same level of theoretical guarantees.   In this paper, we delve into the theoretical and practical aspects of constrained Bayesian optimization, where the objective and constraints can be independently evaluated and are subject to noise. By recognizing that both the objective and constraints can help identify high-confidence regions of interest (ROI), we propose an efficient CBO framework that intersects the ROIs identified from each aspect to determine the general ROI. The ROI, coupled with a novel acquisition function that adaptively balances the optimization of the objective and the identification of feasible regions, enables us to derive rigorous theoretical justifications for its performance. We showcase the efficiency and robustness of our proposed CBO framework through empirical evidence and discuss the fundamental challenge of deriving practical regret bounds for CBO algorithms.

{{</citation>}}


### (105/172) Search-Adaptor: Text Embedding Customization for Information Retrieval (Jinsung Yoon et al., 2023)

{{<citation>}}

Jinsung Yoon, Sercan O Arik, Yanfei Chen, Tomas Pfister. (2023)  
**Search-Adaptor: Text Embedding Customization for Information Retrieval**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, Google, Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08750v1)  

---


**ABSTRACT**  
Text embeddings extracted by pre-trained Large Language Models (LLMs) have significant potential to improve information retrieval and search. Beyond the zero-shot setup in which they are being conventionally used, being able to take advantage of the information from the relevant query-corpus paired data has the power to further boost the LLM capabilities. In this paper, we propose a novel method, Search-Adaptor, for customizing LLMs for information retrieval in an efficient and robust way. Search-Adaptor modifies the original text embedding generated by pre-trained LLMs, and can be integrated with any LLM, including those only available via APIs. On multiple real-world English and multilingual retrieval datasets, we show consistent and significant performance benefits for Search-Adaptor -- e.g., more than 5.2% improvements over the Google Embedding APIs in nDCG@10 averaged over 13 BEIR datasets.

{{</citation>}}


### (106/172) Splicing Up Your Predictions with RNA Contrastive Learning (Philip Fradkin et al., 2023)

{{<citation>}}

Philip Fradkin, Ruian Shi, Bo Wang, Brendan Frey, Leo J. Lee. (2023)  
**Splicing Up Your Predictions with RNA Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-GN  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.08738v1)  

---


**ABSTRACT**  
In the face of rapidly accumulating genomic data, our understanding of the RNA regulatory code remains incomplete. Recent self-supervised methods in other domains have demonstrated the ability to learn rules underlying the data-generating process such as sentence structure in language. Inspired by this, we extend contrastive learning techniques to genomic data by utilizing functional similarities between sequences generated through alternative splicing and gene duplication. Our novel dataset and contrastive objective enable the learning of generalized RNA isoform representations. We validate their utility on downstream tasks such as RNA half-life and mean ribosome load prediction. Our pre-training strategy yields competitive results using linear probing on both tasks, along with up to a two-fold increase in Pearson correlation in low-data conditions. Importantly, our exploration of the learned latent space reveals that our contrastive objective yields semantically meaningful representations, underscoring its potential as a valuable initialization technique for RNA property prediction.

{{</citation>}}


### (107/172) Heterophily-Based Graph Neural Network for Imbalanced Classification (Zirui Liang et al., 2023)

{{<citation>}}

Zirui Liang, Yuntao Li, Tianjin Huang, Akrati Saxena, Yulong Pei, Mykola Pechenizkiy. (2023)  
**Heterophily-Based Graph Neural Network for Imbalanced Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.08725v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have shown promise in addressing graph-related problems, including node classification. However, conventional GNNs assume an even distribution of data across classes, which is often not the case in real-world scenarios, where certain classes are severely underrepresented. This leads to suboptimal performance of standard GNNs on imbalanced graphs. In this paper, we introduce a unique approach that tackles imbalanced classification on graphs by considering graph heterophily. We investigate the intricate relationship between class imbalance and graph heterophily, revealing that minority classes not only exhibit a scarcity of samples but also manifest lower levels of homophily, facilitating the propagation of erroneous information among neighboring nodes. Drawing upon this insight, we propose an efficient method, called Fast Im-GBK, which integrates an imbalance classification strategy with heterophily-aware GNNs to effectively address the class imbalance problem while significantly reducing training time. Our experiments on real-world graphs demonstrate our model's superiority in classification performance and efficiency for node classification tasks compared to existing baselines.

{{</citation>}}


### (108/172) Transformer Choice Net: A Transformer Neural Network for Choice Prediction (Hanzhao Wang et al., 2023)

{{<citation>}}

Hanzhao Wang, Xiaocheng Li, Kalyan Talluri. (2023)  
**Transformer Choice Net: A Transformer Neural Network for Choice Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.08716v1)  

---


**ABSTRACT**  
Discrete-choice models, such as Multinomial Logit, Probit, or Mixed-Logit, are widely used in Marketing, Economics, and Operations Research: given a set of alternatives, the customer is modeled as choosing one of the alternatives to maximize a (latent) utility function. However, extending such models to situations where the customer chooses more than one item (such as in e-commerce shopping) has proven problematic. While one can construct reasonable models of the customer's behavior, estimating such models becomes very challenging because of the combinatorial explosion in the number of possible subsets of items. In this paper we develop a transformer neural network architecture, the Transformer Choice Net, that is suitable for predicting multiple choices. Transformer networks turn out to be especially suitable for this task as they take into account not only the features of the customer and the items but also the context, which in this case could be the assortment as well as the customer's past choices. On a range of benchmark datasets, our architecture shows uniformly superior out-of-sample prediction performance compared to the leading models in the literature, without requiring any custom modeling or tuning for each instance.

{{</citation>}}


### (109/172) Virtual Augmented Reality for Atari Reinforcement Learning (Christian A. Schiller, 2023)

{{<citation>}}

Christian A. Schiller. (2023)  
**Virtual Augmented Reality for Atari Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Google, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08683v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) has achieved significant milestones in the gaming domain, most notably Google DeepMind's AlphaGo defeating human Go champion Ken Jie. This victory was also made possible through the Atari Learning Environment (ALE): The ALE has been foundational in RL research, facilitating significant RL algorithm developments such as AlphaGo and others. In current Atari video game RL research, RL agents' perceptions of its environment is based on raw pixel data from the Atari video game screen with minimal image preprocessing. Contrarily, cutting-edge ML research, external to the Atari video game RL research domain, is focusing on enhancing image perception. A notable example is Meta Research's "Segment Anything Model" (SAM), a foundation model capable of segmenting images without prior training (zero-shot). This paper addresses a novel methodical question: Can state-of-the-art image segmentation models such as SAM improve the performance of RL agents playing Atari video games? The results suggest that SAM can serve as a "virtual augmented reality" for the RL agent, boosting its Atari video game playing performance under certain conditions. Comparing RL agent performance results from raw and augmented pixel inputs provides insight into these conditions. Although this paper was limited by computational constraints, the findings show improved RL agent performance for augmented pixel inputs and can inform broader research agendas in the domain of "virtual augmented reality for video game playing RL agents".

{{</citation>}}


### (110/172) Counting and Algorithmic Generalization with Transformers (Simon Ouellette et al., 2023)

{{<citation>}}

Simon Ouellette, Rolf Pfister, Hansueli Jud. (2023)  
**Counting and Algorithmic Generalization with Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08661v1)  

---


**ABSTRACT**  
Algorithmic generalization in machine learning refers to the ability to learn the underlying algorithm that generates data in a way that generalizes out-of-distribution. This is generally considered a difficult task for most machine learning algorithms. Here, we analyze algorithmic generalization when counting is required, either implicitly or explicitly. We show that standard Transformers are based on architectural decisions that hinder out-of-distribution performance for such tasks. In particular, we discuss the consequences of using layer normalization and of normalizing the attention weights via softmax. With ablation of the problematic operations, we demonstrate that a modified transformer can exhibit a good algorithmic generalization performance on counting while using a very lightweight architecture.

{{</citation>}}


### (111/172) Analyzing Textual Data for Fatality Classification in Afghanistan's Armed Conflicts: A BERT Approach (Hikmatullah Mohammadi et al., 2023)

{{<citation>}}

Hikmatullah Mohammadi, Ziaullah Momand, Parwin Habibi, Nazifa Ramaki, Bibi Storay Fazli, Sayed Zobair Rohany, Iqbal Samsoor. (2023)  
**Analyzing Textual Data for Fatality Classification in Afghanistan's Armed Conflicts: A BERT Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08653v1)  

---


**ABSTRACT**  
Afghanistan has witnessed many armed conflicts throughout history, especially in the past 20 years; these events have had a significant impact on human lives, including military and civilians, with potential fatalities. In this research, we aim to leverage state-of-the-art machine learning techniques to classify the outcomes of Afghanistan armed conflicts to either fatal or non-fatal based on their textual descriptions provided by the Armed Conflict Location & Event Data Project (ACLED) dataset. The dataset contains comprehensive descriptions of armed conflicts in Afghanistan that took place from August 2021 to March 2023. The proposed approach leverages the power of BERT (Bidirectional Encoder Representations from Transformers), a cutting-edge language representation model in natural language processing. The classifier utilizes the raw textual description of an event to estimate the likelihood of the event resulting in a fatality. The model achieved impressive performance on the test set with an accuracy of 98.8%, recall of 98.05%, precision of 99.6%, and an F1 score of 98.82%. These results highlight the model's robustness and indicate its potential impact in various areas such as resource allocation, policymaking, and humanitarian aid efforts in Afghanistan. The model indicates a machine learning-based text classification approach using the ACLED dataset to accurately classify fatality in Afghanistan armed conflicts, achieving robust performance with the BERT model and paving the way for future endeavors in predicting event severity in Afghanistan.

{{</citation>}}


### (112/172) Electrical Grid Anomaly Detection via Tensor Decomposition (Alexander Most et al., 2023)

{{<citation>}}

Alexander Most, Maksim Eren, Nigel Lawrence, Boian Alexandrov. (2023)  
**Electrical Grid Anomaly Detection via Tensor Decomposition**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.08650v1)  

---


**ABSTRACT**  
Supervisory Control and Data Acquisition (SCADA) systems often serve as the nervous system for substations within power grids. These systems facilitate real-time monitoring, data acquisition, control of equipment, and ensure smooth and efficient operation of the substation and its connected devices. Previous work has shown that dimensionality reduction-based approaches, such as Principal Component Analysis (PCA), can be used for accurate identification of anomalies in SCADA systems. While not specifically applied to SCADA, non-negative matrix factorization (NMF) has shown strong results at detecting anomalies in wireless sensor networks. These unsupervised approaches model the normal or expected behavior and detect the unseen types of attacks or anomalies by identifying the events that deviate from the expected behavior. These approaches; however, do not model the complex and multi-dimensional interactions that are naturally present in SCADA systems. Differently, non-negative tensor decomposition is a powerful unsupervised machine learning (ML) method that can model the complex and multi-faceted activity details of SCADA events. In this work, we novelly apply the tensor decomposition method Canonical Polyadic Alternating Poisson Regression (CP-APR) with a probabilistic framework, which has previously shown state-of-the-art anomaly detection results on cyber network data, to identify anomalies in SCADA systems. We showcase that the use of statistical behavior analysis of SCADA communication with tensor decomposition improves the specificity and accuracy of identifying anomalies in electrical grid systems. In our experiments, we model real-world SCADA system data collected from the electrical grid operated by Los Alamos National Laboratory (LANL) which provides transmission and distribution service through a partnership with Los Alamos County, and detect synthetically generated anomalies.

{{</citation>}}


### (113/172) Transformers as Decision Makers: Provable In-Context Reinforcement Learning via Supervised Pretraining (Licong Lin et al., 2023)

{{<citation>}}

Licong Lin, Yu Bai, Song Mei. (2023)  
**Transformers as Decision Makers: Provable In-Context Reinforcement Learning via Supervised Pretraining**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, math-ST, stat-ML, stat-TH  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08566v1)  

---


**ABSTRACT**  
Large transformer models pretrained on offline reinforcement learning datasets have demonstrated remarkable in-context reinforcement learning (ICRL) capabilities, where they can make good decisions when prompted with interaction trajectories from unseen environments. However, when and how transformers can be trained to perform ICRL have not been theoretically well-understood. In particular, it is unclear which reinforcement-learning algorithms transformers can perform in context, and how distribution mismatch in offline training data affects the learned algorithms. This paper provides a theoretical framework that analyzes supervised pretraining for ICRL. This includes two recently proposed training methods -- algorithm distillation and decision-pretrained transformers. First, assuming model realizability, we prove the supervised-pretrained transformer will imitate the conditional expectation of the expert algorithm given the observed trajectory. The generalization error will scale with model capacity and a distribution divergence factor between the expert and offline algorithms. Second, we show transformers with ReLU attention can efficiently approximate near-optimal online reinforcement learning algorithms like LinUCB and Thompson sampling for stochastic linear bandits, and UCB-VI for tabular Markov decision processes. This provides the first quantitative analysis of the ICRL capabilities of transformers pretrained from offline trajectories.

{{</citation>}}


### (114/172) Offline Retraining for Online RL: Decoupled Policy Learning to Mitigate Exploration Bias (Max Sobol Mark et al., 2023)

{{<citation>}}

Max Sobol Mark, Archit Sharma, Fahim Tajwar, Rafael Rafailov, Sergey Levine, Chelsea Finn. (2023)  
**Offline Retraining for Online RL: Decoupled Policy Learning to Mitigate Exploration Bias**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2310.08558v1)  

---


**ABSTRACT**  
It is desirable for policies to optimistically explore new states and behaviors during online reinforcement learning (RL) or fine-tuning, especially when prior offline data does not provide enough state coverage. However, exploration bonuses can bias the learned policy, and our experiments find that naive, yet standard use of such bonuses can fail to recover a performant policy. Concurrently, pessimistic training in offline RL has enabled recovery of performant policies from static datasets. Can we leverage offline RL to recover better policies from online interaction? We make a simple observation that a policy can be trained from scratch on all interaction data with pessimistic objectives, thereby decoupling the policies used for data collection and for evaluation. Specifically, we propose offline retraining, a policy extraction step at the end of online fine-tuning in our Offline-to-Online-to-Offline (OOO) framework for reinforcement learning (RL). An optimistic (exploration) policy is used to interact with the environment, and a separate pessimistic (exploitation) policy is trained on all the observed data for evaluation. Such decoupling can reduce any bias from online interaction (intrinsic rewards, primacy bias) in the evaluation policy, and can allow more exploratory behaviors during online interaction which in turn can generate better data for exploitation. OOO is complementary to several offline-to-online RL and online RL methods, and improves their average performance by 14% to 26% in our fine-tuning experiments, achieves state-of-the-art performance on several environments in the D4RL benchmarks, and improves online RL performance by 165% on two OpenAI gym environments. Further, OOO can enable fine-tuning from incomplete offline datasets where prior methods can fail to recover a performant policy. Implementation: https://github.com/MaxSobolMark/OOO

{{</citation>}}


### (115/172) Cross-Episodic Curriculum for Transformer Agents (Lucy Xiaoyang Shi et al., 2023)

{{<citation>}}

Lucy Xiaoyang Shi, Yunfan Jiang, Jake Grigsby, Linxi "Jim" Fan, Yuke Zhu. (2023)  
**Cross-Episodic Curriculum for Transformer Agents**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.08549v1)  

---


**ABSTRACT**  
We present a new algorithm, Cross-Episodic Curriculum (CEC), to boost the learning efficiency and generalization of Transformer agents. Central to CEC is the placement of cross-episodic experiences into a Transformer's context, which forms the basis of a curriculum. By sequentially structuring online learning trials and mixed-quality demonstrations, CEC constructs curricula that encapsulate learning progression and proficiency increase across episodes. Such synergy combined with the potent pattern recognition capabilities of Transformer models delivers a powerful cross-episodic attention mechanism. The effectiveness of CEC is demonstrated under two representative scenarios: one involving multi-task reinforcement learning with discrete control, such as in DeepMind Lab, where the curriculum captures the learning progression in both individual and progressively complex settings; and the other involving imitation learning with mixed-quality data for continuous control, as seen in RoboMimic, where the curriculum captures the improvement in demonstrators' expertise. In all instances, policies resulting from CEC exhibit superior performance and strong generalization. Code is open-sourced at https://cec-agent.github.io/ to facilitate research on Transformer agent learning.

{{</citation>}}


### (116/172) Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images (Steffen Wolf et al., 2023)

{{<citation>}}

Steffen Wolf, Manan Lalit, Henry Westmacott, Katie McDole, Jan Funke. (2023)  
**Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.08501v1)  

---


**ABSTRACT**  
Segmentation of objects in microscopy images is required for many biomedical applications. We introduce object-centric embeddings (OCEs), which embed image patches such that the spatial offsets between patches cropped from the same object are preserved. Those learnt embeddings can be used to delineate individual objects and thus obtain instance segmentations. Here, we show theoretically that, under assumptions commonly found in microscopy images, OCEs can be learnt through a self-supervised task that predicts the spatial offset between image patches. Together, this forms an unsupervised cell instance segmentation method which we evaluate on nine diverse large-scale microscopy datasets. Segmentations obtained with our method lead to substantially improved results, compared to state-of-the-art baselines on six out of nine datasets, and perform on par on the remaining three datasets. If ground-truth annotations are available, our method serves as an excellent starting point for supervised training, reducing the required amount of ground-truth needed by one order of magnitude, thus substantially increasing the practical applicability of our method. Source code is available at https://github.com/funkelab/cellulus.

{{</citation>}}


### (117/172) A Survey on Heterogeneous Transfer Learning (Runxue Bao et al., 2023)

{{<citation>}}

Runxue Bao, Yiming Sun, Yuhe Gao, Jindong Wang, Qiang Yang, Haifeng Chen, Zhi-Hong Mao, Xing Xie, Ye Ye. (2023)  
**A Survey on Heterogeneous Transfer Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Computer Vision, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.08459v1)  

---


**ABSTRACT**  
The application of transfer learning, an approach utilizing knowledge from a source domain to enhance model performance in a target domain, has seen a tremendous rise in recent years, underpinning many real-world scenarios. The key to its success lies in the shared common knowledge between the domains, a prerequisite in most transfer learning methodologies. These methods typically presuppose identical feature spaces and label spaces in both domains, known as homogeneous transfer learning, which, however, is not always a practical assumption. Oftentimes, the source and target domains vary in feature spaces, data distributions, and label spaces, making it challenging or costly to secure source domain data with identical feature and label spaces as the target domain. Arbitrary elimination of these differences is not always feasible or optimal. Thus, heterogeneous transfer learning, acknowledging and dealing with such disparities, has emerged as a promising approach for a variety of tasks. Despite the existence of a survey in 2017 on this topic, the fast-paced advances post-2017 necessitate an updated, in-depth review. We therefore present a comprehensive survey of recent developments in heterogeneous transfer learning methods, offering a systematic guide for future research. Our paper reviews methodologies for diverse learning scenarios, discusses the limitations of current studies, and covers various application contexts, including Natural Language Processing, Computer Vision, Multimodality, and Biomedicine, to foster a deeper understanding and spur future research.

{{</citation>}}


### (118/172) Towards Robust Multi-Modal Reasoning via Model Selection (Xiangyan Liu et al., 2023)

{{<citation>}}

Xiangyan Liu, Rongxue Li, Wei Ji, Tao Lin. (2023)  
**Towards Robust Multi-Modal Reasoning via Model Selection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.08446v1)  

---


**ABSTRACT**  
The reasoning capabilities of LLM (Large Language Model) are widely acknowledged in recent research, inspiring studies on tool learning and autonomous agents. LLM serves as the "brain" of agent, orchestrating multiple tools for collaborative multi-step task solving. Unlike methods invoking tools like calculators or weather APIs for straightforward tasks, multi-modal agents excel by integrating diverse AI models for complex challenges. However, current multi-modal agents neglect the significance of model selection: they primarily focus on the planning and execution phases, and will only invoke predefined task-specific models for each subtask, making the execution fragile. Meanwhile, other traditional model selection methods are either incompatible with or suboptimal for the multi-modal agent scenarios, due to ignorance of dependencies among subtasks arising by multi-step reasoning.   To this end, we identify the key challenges therein and propose the $\textit{M}^3$ framework as a plug-in with negligible runtime overhead at test-time. This framework improves model selection and bolsters the robustness of multi-modal agents in multi-step reasoning. In the absence of suitable benchmarks, we create MS-GQA, a new dataset specifically designed to investigate the model selection challenge in multi-modal agents. Our experiments reveal that our framework enables dynamic model selection, considering both user inputs and subtask dependencies, thereby robustifying the overall reasoning process. Our code and benchmark: https://github.com/LINs-lab/M3.

{{</citation>}}


### (119/172) Differentially Private Non-convex Learning for Multi-layer Neural Networks (Hanpu Shen et al., 2023)

{{<citation>}}

Hanpu Shen, Cheng-Long Wang, Zihang Xiang, Yiming Ying, Di Wang. (2023)  
**Differentially Private Non-convex Learning for Multi-layer Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG, stat-ML  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2310.08425v1)  

---


**ABSTRACT**  
This paper focuses on the problem of Differentially Private Stochastic Optimization for (multi-layer) fully connected neural networks with a single output node. In the first part, we examine cases with no hidden nodes, specifically focusing on Generalized Linear Models (GLMs). We investigate the well-specific model where the random noise possesses a zero mean, and the link function is both bounded and Lipschitz continuous. We propose several algorithms and our analysis demonstrates the feasibility of achieving an excess population risk that remains invariant to the data dimension. We also delve into the scenario involving the ReLU link function, and our findings mirror those of the bounded link function. We conclude this section by contrasting well-specified and misspecified models, using ReLU regression as a representative example.   In the second part of the paper, we extend our ideas to two-layer neural networks with sigmoid or ReLU activation functions in the well-specified model. In the third part, we study the theoretical guarantees of DP-SGD in Abadi et al. (2016) for fully connected multi-layer neural networks. By utilizing recent advances in Neural Tangent Kernel theory, we provide the first excess population risk when both the sample size and the width of the network are sufficiently large. Additionally, we discuss the role of some parameters in DP-SGD regarding their utility, both theoretically and empirically.

{{</citation>}}


### (120/172) Jailbreaking Black Box Large Language Models in Twenty Queries (Patrick Chao et al., 2023)

{{<citation>}}

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong. (2023)  
**Jailbreaking Black Box Large Language Models in Twenty Queries**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, GPT, GPT-3.5, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2310.08419v1)  

---


**ABSTRACT**  
There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and PaLM-2.

{{</citation>}}


### (121/172) Neural Diffusion Models (Grigory Bartosh et al., 2023)

{{<citation>}}

Grigory Bartosh, Dmitry Vetrov, Christian A. Naesseth. (2023)  
**Neural Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08337v1)  

---


**ABSTRACT**  
Diffusion models have shown remarkable performance on many generative tasks. Despite recent success, most diffusion models are restricted in that they only allow linear transformation of the data distribution. In contrast, broader family of transformations can potentially help train generative distributions more efficiently, simplifying the reverse process and closing the gap between the true negative log-likelihood and the variational approximation. In this paper, we present Neural Diffusion Models (NDMs), a generalization of conventional diffusion models that enables defining and learning time-dependent non-linear transformations of data. We show how to optimise NDMs using a variational bound in a simulation-free setting. Moreover, we derive a time-continuous formulation of NDMs, which allows fast and reliable inference using off-the-shelf numerical ODE and SDE solvers. Finally, we demonstrate the utility of NDMs with learnable transformations through experiments on standard image generation benchmarks, including CIFAR-10, downsampled versions of ImageNet and CelebA-HQ. NDMs outperform conventional diffusion models in terms of likelihood and produce high-quality samples.

{{</citation>}}


### (122/172) Defending Our Privacy With Backdoors (Dominik Hintersdorf et al., 2023)

{{<citation>}}

Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting. (2023)  
**Defending Our Privacy With Backdoors**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08320v1)  

---


**ABSTRACT**  
The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

{{</citation>}}


### (123/172) Lag-Llama: Towards Foundation Models for Time Series Forecasting (Kashif Rasul et al., 2023)

{{<citation>}}

Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Biloš, Hena Ghonia, Nadhir Vincent Hassen, Anderson Schneider, Sahil Garg, Alexandre Drouin, Nicolas Chapados, Yuriy Nevmyvaka, Irina Rish. (2023)  
**Lag-Llama: Towards Foundation Models for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.08278v1)  

---


**ABSTRACT**  
Aiming to build foundation models for time-series forecasting and study their scaling behavior, we present here our work-in-progress on Lag-Llama, a general-purpose univariate probabilistic time-series forecasting model trained on a large collection of time-series data. The model shows good zero-shot prediction capabilities on unseen "out-of-distribution" time-series datasets, outperforming supervised baselines. We use smoothly broken power-laws to fit and predict model scaling behavior. The open source code is made available at https://github.com/kashif/pytorch-transformer-ts.

{{</citation>}}


### (124/172) MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning (Zeyuan Ma et al., 2023)

{{<citation>}}

Zeyuan Ma, Hongshu Guo, Jiacheng Chen, Zhenrui Li, Guojun Peng, Yue-Jiao Gong, Yining Ma, Zhiguang Cao. (2023)  
**MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08252v1)  

---


**ABSTRACT**  
Recently, Meta-Black-Box Optimization with Reinforcement Learning (MetaBBO-RL) has showcased the power of leveraging RL at the meta-level to mitigate manual fine-tuning of low-level black-box optimizers. However, this field is hindered by the lack of a unified benchmark. To fill this gap, we introduce MetaBox, the first benchmark platform expressly tailored for developing and evaluating MetaBBO-RL methods. MetaBox offers a flexible algorithmic template that allows users to effortlessly implement their unique designs within the platform. Moreover, it provides a broad spectrum of over 300 problem instances, collected from synthetic to realistic scenarios, and an extensive library of 19 baseline methods, including both traditional black-box optimizers and recent MetaBBO-RL methods. Besides, MetaBox introduces three standardized performance metrics, enabling a more thorough assessment of the methods. In a bid to illustrate the utility of MetaBox for facilitating rigorous evaluation and in-depth analysis, we carry out a wide-ranging benchmarking study on existing MetaBBO-RL methods. Our MetaBox is open-source and accessible at: https://github.com/GMC-DRL/MetaBox.

{{</citation>}}


### (125/172) Beyond Traditional DoE: Deep Reinforcement Learning for Optimizing Experiments in Model Identification of Battery Dynamics (Gokhan Budan et al., 2023)

{{<citation>}}

Gokhan Budan, Francesca Damiani, Can Kurtulus, N. Kemal Ure. (2023)  
**Beyond Traditional DoE: Deep Reinforcement Learning for Optimizing Experiments in Model Identification of Battery Dynamics**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08198v1)  

---


**ABSTRACT**  
Model identification of battery dynamics is a central problem in energy research; many energy management systems and design processes rely on accurate battery models for efficiency optimization. The standard methodology for battery modelling is traditional design of experiments (DoE), where the battery dynamics are excited with many different current profiles and the measured outputs are used to estimate the system dynamics. However, although it is possible to obtain useful models with the traditional approach, the process is time consuming and expensive because of the need to sweep many different current-profile configurations. In the present work, a novel DoE approach is developed based on deep reinforcement learning, which alters the configuration of the experiments on the fly based on the statistics of past experiments. Instead of sticking to a library of predefined current profiles, the proposed approach modifies the current profiles dynamically by updating the output space covered by past measurements, hence only the current profiles that are informative for future experiments are applied. Simulations and real experiments are used to show that the proposed approach gives models that are as accurate as those obtained with traditional DoE but by using 85\% less resources.

{{</citation>}}


### (126/172) Infinite Width Graph Neural Networks for Node Regression/ Classification (Yunus Cobanoglu, 2023)

{{<citation>}}

Yunus Cobanoglu. (2023)  
**Infinite Width Graph Neural Networks for Node Regression/ Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.08176v1)  

---


**ABSTRACT**  
This work analyzes Graph Neural Networks, a generalization of Fully-Connected Deep Neural Nets on Graph structured data, when their width, that is the number of nodes in each fullyconnected layer is increasing to infinity. Infinite Width Neural Networks are connecting Deep Learning to Gaussian Processes and Kernels, both Machine Learning Frameworks with long traditions and extensive theoretical foundations. Gaussian Processes and Kernels have much less hyperparameters then Neural Networks and can be used for uncertainty estimation, making them more user friendly for applications. This works extends the increasing amount of research connecting Gaussian Processes and Kernels to Neural Networks. The Kernel and Gaussian Process closed forms are derived for a variety of architectures, namely the standard Graph Neural Network, the Graph Neural Network with Skip-Concatenate Connections and the Graph Attention Neural Network. All architectures are evaluated on a variety of datasets on the task of transductive Node Regression and Classification. Additionally, a Spectral Sparsification method known as Effective Resistance is used to improve runtime and memory requirements. Extending the setting to inductive graph learning tasks (Graph Regression/ Classification) is straightforward and is briefly discussed in 3.5.

{{</citation>}}


### (127/172) Interpreting Reward Models in RLHF-Tuned Language Models Using Sparse Autoencoders (Luke Marks et al., 2023)

{{<citation>}}

Luke Marks, Amir Abdullah, Luna Mendez, Rauno Arike, Philip Torr, Fazl Barez. (2023)  
**Interpreting Reward Models in RLHF-Tuned Language Models Using Sparse Autoencoders**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08164v1)  

---


**ABSTRACT**  
Large language models (LLMs) aligned to human preferences via reinforcement learning from human feedback (RLHF) underpin many commercial applications. However, how RLHF impacts LLM internals remains opaque. We propose a novel method to interpret learned reward functions in RLHF-tuned LLMs using sparse autoencoders. Our approach trains autoencoder sets on activations from a base LLM and its RLHF-tuned version. By comparing autoencoder hidden spaces, we identify unique features that reflect the accuracy of the learned reward model. To quantify this, we construct a scenario where the tuned LLM learns token-reward mappings to maximize reward. This is the first application of sparse autoencoders for interpreting learned rewards and broadly inspecting reward learning in LLMs. Our method provides an abstract approximation of reward integrity. This presents a promising technique for ensuring alignment between specified objectives and model behaviors.

{{</citation>}}


### (128/172) Open-Set Knowledge-Based Visual Question Answering with Inference Paths (Jingru Gan et al., 2023)

{{<citation>}}

Jingru Gan, Xinzhe Han, Shuhui Wang, Qingming Huang. (2023)  
**Open-Set Knowledge-Based Visual Question Answering with Inference Paths**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.08148v1)  

---


**ABSTRACT**  
Given an image and an associated textual question, the purpose of Knowledge-Based Visual Question Answering (KB-VQA) is to provide a correct answer to the question with the aid of external knowledge bases. Prior KB-VQA models are usually formulated as a retriever-classifier framework, where a pre-trained retriever extracts textual or visual information from knowledge graphs and then makes a prediction among the candidates. Despite promising progress, there are two drawbacks with existing models. Firstly, modeling question-answering as multi-class classification limits the answer space to a preset corpus and lacks the ability of flexible reasoning. Secondly, the classifier merely consider "what is the answer" without "how to get the answer", which cannot ground the answer to explicit reasoning paths. In this paper, we confront the challenge of \emph{explainable open-set} KB-VQA, where the system is required to answer questions with entities at wild and retain an explainable reasoning path. To resolve the aforementioned issues, we propose a new retriever-ranker paradigm of KB-VQA, Graph pATH rankER (GATHER for brevity). Specifically, it contains graph constructing, pruning, and path-level ranking, which not only retrieves accurate answers but also provides inference paths that explain the reasoning process. To comprehensively evaluate our model, we reformulate the benchmark dataset OK-VQA with manually corrected entity-level annotations and release it as ConceptVQA. Extensive experiments on real-world questions demonstrate that our framework is not only able to perform open-set question answering across the whole knowledge base but provide explicit reasoning path.

{{</citation>}}


### (129/172) Counterfactual Explanations for Time Series Forecasting (Zhendong Wang et al., 2023)

{{<citation>}}

Zhendong Wang, Ioanna Miliou, Isak Samsten, Panagiotis Papapetrou. (2023)  
**Counterfactual Explanations for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.08137v1)  

---


**ABSTRACT**  
Among recent developments in time series forecasting methods, deep forecasting models have gained popularity as they can utilize hidden feature patterns in time series to improve forecasting performance. Nevertheless, the majority of current deep forecasting models are opaque, hence making it challenging to interpret the results. While counterfactual explanations have been extensively employed as a post-hoc approach for explaining classification models, their application to forecasting models still remains underexplored. In this paper, we formulate the novel problem of counterfactual generation for time series forecasting, and propose an algorithm, called ForecastCF, that solves the problem by applying gradient-based perturbations to the original time series. ForecastCF guides the perturbations by applying constraints to the forecasted values to obtain desired prediction outcomes. We experimentally evaluate ForecastCF using four state-of-the-art deep model architectures and compare to two baselines. Our results show that ForecastCF outperforms the baseline in terms of counterfactual validity and data manifold closeness. Overall, our findings suggest that ForecastCF can generate meaningful and relevant counterfactual explanations for various forecasting tasks.

{{</citation>}}


### (130/172) ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets (Tobias Schimanski et al., 2023)

{{<citation>}}

Tobias Schimanski, Julia Bingler, Camilla Hyslop, Mathias Kraus, Markus Leippold. (2023)  
**ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.08096v1)  

---


**ABSTRACT**  
Public and private actors struggle to assess the vast amounts of information about sustainability commitments made by various institutions. To address this problem, we create a novel tool for automatically detecting corporate, national, and regional net zero and reduction targets in three steps. First, we introduce an expert-annotated data set with 3.5K text samples. Second, we train and release ClimateBERT-NetZero, a natural language classifier to detect whether a text contains a net zero or reduction target. Third, we showcase its analysis potential with two use cases: We first demonstrate how ClimateBERT-NetZero can be combined with conventional question-answering (Q&A) models to analyze the ambitions displayed in net zero and reduction targets. Furthermore, we employ the ClimateBERT-NetZero model on quarterly earning call transcripts and outline how communication patterns evolve over time. Our experiments demonstrate promising pathways for extracting and analyzing net zero and emission reduction targets at scale.

{{</citation>}}


### (131/172) Samples on Thin Ice: Re-Evaluating Adversarial Pruning of Neural Networks (Giorgio Piras et al., 2023)

{{<citation>}}

Giorgio Piras, Maura Pintor, Ambra Demontis, Battista Biggio. (2023)  
**Samples on Thin Ice: Re-Evaluating Adversarial Pruning of Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2310.08073v1)  

---


**ABSTRACT**  
Neural network pruning has shown to be an effective technique for reducing the network size, trading desirable properties like generalization and robustness to adversarial attacks for higher sparsity. Recent work has claimed that adversarial pruning methods can produce sparse networks while also preserving robustness to adversarial examples. In this work, we first re-evaluate three state-of-the-art adversarial pruning methods, showing that their robustness was indeed overestimated. We then compare pruned and dense versions of the same models, discovering that samples on thin ice, i.e., closer to the unpruned model's decision boundary, are typically misclassified after pruning. We conclude by discussing how this intuition may lead to designing more effective adversarial pruning methods in future work.

{{</citation>}}


### (132/172) Learning from Label Proportions: Bootstrapping Supervised Learners via Belief Propagation (Shreyas Havaldar et al., 2023)

{{<citation>}}

Shreyas Havaldar, Navodita Sharma, Shubhi Sareen, Karthikeyan Shanmugam, Aravindan Raghuveer. (2023)  
**Learning from Label Proportions: Bootstrapping Supervised Learners via Belief Propagation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.08056v1)  

---


**ABSTRACT**  
Learning from Label Proportions (LLP) is a learning problem where only aggregate level labels are available for groups of instances, called bags, during training, and the aim is to get the best performance at the instance-level on the test data. This setting arises in domains like advertising and medicine due to privacy considerations. We propose a novel algorithmic framework for this problem that iteratively performs two main steps. For the first step (Pseudo Labeling) in every iteration, we define a Gibbs distribution over binary instance labels that incorporates a) covariate information through the constraint that instances with similar covariates should have similar labels and b) the bag level aggregated label. We then use Belief Propagation (BP) to marginalize the Gibbs distribution to obtain pseudo labels. In the second step (Embedding Refinement), we use the pseudo labels to provide supervision for a learner that yields a better embedding. Further, we iterate on the two steps again by using the second step's embeddings as new covariates for the next iteration. In the final iteration, a classifier is trained using the pseudo labels. Our algorithm displays strong gains against several SOTA baselines (up to 15%) for the LLP Binary Classification problem on various dataset types - tabular and Image. We achieve these improvements with minimal computational overhead above standard supervised learning due to Belief Propagation, for large bag sizes, even for a million samples.

{{</citation>}}


### (133/172) Continual Learning via Manifold Expansion Replay (Zihao Xu et al., 2023)

{{<citation>}}

Zihao Xu, Xuan Tang, Yufei Shi, Jianfeng Zhang, Jian Yang, Mingsong Chen, Xian Wei. (2023)  
**Continual Learning via Manifold Expansion Replay**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-IR, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08038v1)  

---


**ABSTRACT**  
In continual learning, the learner learns multiple tasks in sequence, with data being acquired only once for each task. Catastrophic forgetting is a major challenge to continual learning. To reduce forgetting, some existing rehearsal-based methods use episodic memory to replay samples of previous tasks. However, in the process of knowledge integration when learning a new task, this strategy also suffers from catastrophic forgetting due to an imbalance between old and new knowledge. To address this problem, we propose a novel replay strategy called Manifold Expansion Replay (MaER). We argue that expanding the implicit manifold of the knowledge representation in the episodic memory helps to improve the robustness and expressiveness of the model. To this end, we propose a greedy strategy to keep increasing the diameter of the implicit manifold represented by the knowledge in the buffer during memory management. In addition, we introduce Wasserstein distance instead of cross entropy as distillation loss to preserve previous knowledge. With extensive experimental validation on MNIST, CIFAR10, CIFAR100, and TinyImageNet, we show that the proposed method significantly improves the accuracy in continual learning setup, outperforming the state of the arts.

{{</citation>}}


### (134/172) LEMON: Lossless model expansion (Yite Wang et al., 2023)

{{<citation>}}

Yite Wang, Jiahao Su, Hanlin Lu, Cong Xie, Tianyi Liu, Jianbo Yuan, Haibin Lin, Ruoyu Sun, Hongxia Yang. (2023)  
**LEMON: Lossless model expansion**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07999v1)  

---


**ABSTRACT**  
Scaling of deep neural networks, especially Transformers, is pivotal for their surging performance and has further led to the emergence of sophisticated reasoning capabilities in foundation models. Such scaling generally requires training large models from scratch with random initialization, failing to leverage the knowledge acquired by their smaller counterparts, which are already resource-intensive to obtain. To tackle this inefficiency, we present $\textbf{L}$ossl$\textbf{E}$ss $\textbf{MO}$del Expansio$\textbf{N}$ (LEMON), a recipe to initialize scaled models using the weights of their smaller but pre-trained counterparts. This is followed by model training with an optimized learning rate scheduler tailored explicitly for the scaled models, substantially reducing the training time compared to training from scratch. Notably, LEMON is versatile, ensuring compatibility with various network structures, including models like Vision Transformers and BERT. Our empirical results demonstrate that LEMON reduces computational costs by 56.7% for Vision Transformers and 33.2% for BERT when compared to training from scratch.

{{</citation>}}


### (135/172) Reinforcement Learning of Display Transfer Robots in Glass Flow Control Systems: A Physical Simulation-Based Approach (Hwajong Lee et al., 2023)

{{<citation>}}

Hwajong Lee, Chan Kim, Seong-Woo Kim. (2023)  
**Reinforcement Learning of Display Transfer Robots in Glass Flow Control Systems: A Physical Simulation-Based Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07981v1)  

---


**ABSTRACT**  
A flow control system is a critical concept for increasing the production capacity of manufacturing systems. To solve the scheduling optimization problem related to the flow control with the aim of improving productivity, existing methods depend on a heuristic design by domain human experts. Therefore, the methods require correction, monitoring, and verification by using real equipment. As system designs increase in complexity, the monitoring time increases, which decreases the probability of arriving at the optimal design. As an alternative approach to the heuristic design of flow control systems, the use of deep reinforcement learning to solve the scheduling optimization problem has been considered. Although the existing research on reinforcement learning has yielded excellent performance in some areas, the applicability of the results to actual FAB such as display and semiconductor manufacturing processes is not evident so far. To this end, we propose a method to implement a physical simulation environment and devise a feasible flow control system design using a transfer robot in display manufacturing through reinforcement learning. We present a model and parameter setting to build a virtual environment for different display transfer robots, and training methods of reinforcement learning on the environment to obtain an optimal scheduling of glass flow control systems. Its feasibility was verified by using different types of robots used in the actual process.

{{</citation>}}


### (136/172) GRASP: Accelerating Shortest Path Attacks via Graph Attention (Zohair Shafi. Benjamin A. Miller et al., 2023)

{{<citation>}}

Zohair Shafi. Benjamin A. Miller, Ayan Chatterjee, Tina Eliassi-Rad, Rajmonda S. Caceres. (2023)  
**GRASP: Accelerating Shortest Path Attacks via Graph Attention**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.07980v1)  

---


**ABSTRACT**  
Recent advances in machine learning (ML) have shown promise in aiding and accelerating classical combinatorial optimization algorithms. ML-based speed ups that aim to learn in an end to end manner (i.e., directly output the solution) tend to trade off run time with solution quality. Therefore, solutions that are able to accelerate existing solvers while maintaining their performance guarantees, are of great interest. We consider an APX-hard problem, where an adversary aims to attack shortest paths in a graph by removing the minimum number of edges. We propose the GRASP algorithm: Graph Attention Accelerated Shortest Path Attack, an ML aided optimization algorithm that achieves run times up to 10x faster, while maintaining the quality of solution generated. GRASP uses a graph attention network to identify a smaller subgraph containing the combinatorial solution, thus effectively reducing the input problem size. Additionally, we demonstrate how careful representation of the input graph, including node features that correlate well with the optimization task, can highlight important structure in the optimization solution.

{{</citation>}}


### (137/172) Graph-SCP: Accelerating Set Cover Problems with Graph Neural Networks (Zohair Shafi et al., 2023)

{{<citation>}}

Zohair Shafi, Benjamin A. Miller, Tina Eliassi-Rad, Rajmonda S. Caceres. (2023)  
**Graph-SCP: Accelerating Set Cover Problems with Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-DM, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.07979v1)  

---


**ABSTRACT**  
Machine learning (ML) approaches are increasingly being used to accelerate combinatorial optimization (CO) problems. We look specifically at the Set Cover Problem (SCP) and propose Graph-SCP, a graph neural network method that can augment existing optimization solvers by learning to identify a much smaller sub-problem that contains the solution space. We evaluate the performance of Graph-SCP on synthetic weighted and unweighted SCP instances with diverse problem characteristics and complexities, and on instances from the OR Library, a canonical benchmark for SCP. We show that Graph-SCP reduces the problem size by 30-70% and achieves run time speedups up to~25x when compared to commercial solvers (Gurobi). Given a desired optimality threshold, Graph-SCP will improve upon it or even achieve 100% optimality. This is in contrast to fast greedy solutions that significantly compromise solution quality to achieve guaranteed polynomial run time. Graph-SCP can generalize to larger problem sizes and can be used with other conventional or ML-augmented CO solvers to lead to potential additional run time improvement.

{{</citation>}}


## cs.SD (2)



### (138/172) CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models (Sreyan Ghosh et al., 2023)

{{<citation>}}

Sreyan Ghosh, Ashish Seth, Sonal Kumar, Utkarsh Tyagi, Chandra Kiran Evuru, S. Ramaneswaran, S. Sakshi, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha. (2023)  
**CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.08753v1)  

---


**ABSTRACT**  
A fundamental characteristic of audio is its compositional nature. Audio-language models (ALMs) trained using a contrastive approach (e.g., CLAP) that learns a shared representation between audio and language modalities have improved performance in many downstream applications, including zero-shot audio classification, audio retrieval, etc. However, the ability of these models to effectively perform compositional reasoning remains largely unexplored and necessitates additional research. In this paper, we propose CompA, a collection of two expert-annotated benchmarks with a majority of real-world audio samples, to evaluate compositional reasoning in ALMs. Our proposed CompA-order evaluates how well an ALM understands the order or occurrence of acoustic events in audio, and CompA-attribute evaluates attribute binding of acoustic events. An instance from either benchmark consists of two audio-caption pairs, where both audios have the same acoustic events but with different compositions. An ALM is evaluated on how well it matches the right audio to the right caption. Using this benchmark, we first show that current ALMs perform only marginally better than random chance, thereby struggling with compositional reasoning. Next, we propose CompA-CLAP, where we fine-tune CLAP using a novel learning method to improve its compositional reasoning abilities. To train CompA-CLAP, we first propose improvements to contrastive training with composition-aware hard negatives, allowing for more focused training. Next, we propose a novel modular contrastive loss that helps the model learn fine-grained compositional understanding and overcomes the acute scarcity of openly available compositional audios. CompA-CLAP significantly improves over all our baseline models on the CompA benchmark, indicating its superior compositional reasoning capabilities.

{{</citation>}}


### (139/172) Impact of time and note duration tokenizations on deep learning symbolic music modeling (Nathan Fradet et al., 2023)

{{<citation>}}

Nathan Fradet, Nicolas Gutowski, Fabien Chhel, Jean-Pierre Briot. (2023)  
**Impact of time and note duration tokenizations on deep learning symbolic music modeling**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Information Retrieval, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08497v1)  

---


**ABSTRACT**  
Symbolic music is widely used in various deep learning tasks, including generation, transcription, synthesis, and Music Information Retrieval (MIR). It is mostly employed with discrete models like Transformers, which require music to be tokenized, i.e., formatted into sequences of distinct elements called tokens. Tokenization can be performed in different ways. As Transformer can struggle at reasoning, but capture more easily explicit information, it is important to study how the way the information is represented for such model impact their performances. In this work, we analyze the common tokenization methods and experiment with time and note duration representations. We compare the performances of these two impactful criteria on several tasks, including composer and emotion classification, music generation, and sequence representation learning. We demonstrate that explicit information leads to better results depending on the task.

{{</citation>}}


## cs.SI (2)



### (140/172) Biased news sharing and partisan polarization on social media (Sofía M del Pozo et al., 2023)

{{<citation>}}

Sofía M del Pozo, Sebastián Pinto, Matteo Serafino, Lucio Garcia, Hernán A Makse, Pablo Balenzuela. (2023)  
**Biased news sharing and partisan polarization on social media**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.08701v1)  

---


**ABSTRACT**  
In the ever-connected digital landscape, news dissemination on social media platforms serves as a vital source of information for the public. However, this flow of information is far from unbiased. It is deeply influenced by the political inclinations of the users who share news as well as the inherent biases present in the news outlets themselves. These biases in news consumption play a significant role in the creation of echo chambers and the reinforcement of beliefs. This phenomenon, in turn, influences the voting intentions of the population during critical electoral periods. In this study, we use a metric called "Sentiment Bias", a tool designed to classify news outlets according to their biases. We explore the impact of this metric on various levels, ranging from news outlets to individual user biases. Our metric, while simple, unveils a well-known trend: users prefer news aligning with their political beliefs. Its power lies in extending this insight to specific topics. Users consistently share articles related to subjects that echo their favored candidates, illuminating a deeper layer of political alignment in online discourse.

{{</citation>}}


### (141/172) CODY: A graph-based framework for the analysis of COnversation DYnamics in online social networks (John Ziegler et al., 2023)

{{<citation>}}

John Ziegler, Fabian Kneissl, Michael Gertz. (2023)  
**CODY: A graph-based framework for the analysis of COnversation DYnamics in online social networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.08140v1)  

---


**ABSTRACT**  
Conversations are an integral part of online social media, and gaining insights into these conversations is of significant value for many commercial as well as academic use cases. From a computational perspective, however, analyzing conversation data is complex, and numerous aspects must be considered. Next to the structure of conversations, the discussed content - as well as their dynamics - have to be taken into account. Still, most existing modelling and analysis approaches focus only on one of these aspects and, in particular, lack the capability to investigate the temporal evolution of a conversation. To address these shortcomings, in this work, we present CODY, a content-aware, graph-based framework to study the dynamics of online conversations along multiple dimensions. Its capabilities are extensively demonstrated by conducting three experiments based on a large conversation dataset from the German political Twittersphere. First, the posting activity across the lifetime of conversations is examined. We find that posting activity follows an exponential saturation pattern. Based on this activity model, we develop a volume-based sampling method to study conversation dynamics using temporal network snapshots. In a second experiment, we focus on the evolution of a conversation's structure and leverage a novel metric, the temporal Wiener index, for that. Results indicate that as conversations progress, a conversation's structure tends to be less sprawling and more centered around the original seed post. Furthermore, focusing on the dynamics of content in conversations, the evolution of hashtag usage within conversations is studied. Initially used hashtags do not necessarily keep their dominant prevalence throughout the lifetime of a conversation. Instead, various "hashtag hijacking" scenarios are found.

{{</citation>}}


## cs.SE (3)



### (142/172) CoLadder: Supporting Programmers with Hierarchical Code Generation in Multi-Level Abstraction (Ryan Yen et al., 2023)

{{<citation>}}

Ryan Yen, Jiawen Zhu, Sangho Suh, Haijun Xia, Jian Zhao. (2023)  
**CoLadder: Supporting Programmers with Hierarchical Code Generation in Multi-Level Abstraction**  

---
Primary Category: cs.SE  
Categories: cs-HC, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08699v1)  

---


**ABSTRACT**  
Programmers increasingly rely on Large Language Models (LLMs) for code generation. However, they now have to deal with issues like having to constantly switch between generating and verifying code, caused by misalignment between programmers' prompts and the generated code. Unfortunately, current LLM-driven code assistants provide insufficient support during the prompt authoring process to help programmers tackle these challenges emerging from the new workflow. To address these challenges, we employed an iterative design process to understand programmers' strategies when programming with LLMs. Based on our findings, we developed CoLadder, a system that assists programmers by enabling hierarchical task decomposition, incremental code generation, and verification of results during prompt authoring. A user study with 12 experienced programmers showed that CoLadder is effective in helping programmers externalize their mental models flexibly, improving their ability to navigate and edit code across various abstraction levels, from initial intent to final code implementation.

{{</citation>}}


### (143/172) MCRepair: Multi-Chunk Program Repair via Patch Optimization with Buggy Block (Jisung Kim et al., 2023)

{{<citation>}}

Jisung Kim, Byeongjung Lee. (2023)  
**MCRepair: Multi-Chunk Program Repair via Patch Optimization with Buggy Block**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.08157v1)  

---


**ABSTRACT**  
Automated program repair (APR) is a technology that identifies and repairs bugs automatically. However, repairing multi-chunk bugs remains a long-standing and challenging problem because an APR technique must consider dependencies and then reduce the large patch space. In addition, little is known about how to combine individual candidate patches even though multi-chunk bugs require combinations. Therefore, we propose a novel APR technique called multi-code repair (MCRepair), which applies a buggy block, patch optimization, and CodeBERT to target multi-chunk bugs. A buggy block is a novel method that binds buggy chunks into a multi-buggy chunk and preprocesses the chunk with its buggy contexts for patch space reduction and dependency problems. Patch optimization is a novel strategy that effectively combines the generated candidate patches with patch space reduction. In addition, CodeBERT, a BERT for source code datasets, is fine-tuned to address the lack of datasets and out-of-vocabulary problems. We conducted several experiments to evaluate our approach on six project modules of Defects4J. In the experiments using Defects4J, MCRepair repaired 65 bugs, including 21 multi-chunk bugs. Moreover, it fixed 18 unique bugs, including eight multi-chunk bugs, and improved 40 to 250 percent performance than the baselines.

{{</citation>}}


### (144/172) Towards Causal Deep Learning for Vulnerability Detection (Md Mahbubur Rahman et al., 2023)

{{<citation>}}

Md Mahbubur Rahman, Ira Ceka, Chengzhi Mao, Saikat Chakraborty, Baishakhi Ray, Wei Le. (2023)  
**Towards Causal Deep Learning for Vulnerability Detection**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-LG, cs-SE, cs.SE, stat-ME  
Keywords: Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2310.07958v2)  

---


**ABSTRACT**  
Deep learning vulnerability detection has shown promising results in recent years. However, an important challenge that still blocks it from being very useful in practice is that the model is not robust under perturbation and it cannot generalize well over the out-of-distribution (OOD) data, e.g., applying a trained model to unseen projects in real world. We hypothesize that this is because the model learned non-robust features, e.g., variable names, that have spurious correlations with labels. When the perturbed and OOD datasets no longer have the same spurious features, the model prediction fails. To address the challenge, in this paper, we introduced causality into deep learning vulnerability detection. Our approach CausalVul consists of two phases. First, we designed novel perturbations to discover spurious features that the model may use to make predictions. Second, we applied the causal learning algorithms, specifically, do-calculus, on top of existing deep learning models to systematically remove the use of spurious features and thus promote causal based prediction. Our results show that CausalVul consistently improved the model accuracy, robustness and OOD performance for all the state-of-the-art models and datasets we experimented. To the best of our knowledge, this is the first work that introduces do calculus based causal learning to software engineering models and shows it's indeed useful for improving the model accuracy, robustness and generalization. Our replication package is located at https://figshare.com/s/0ffda320dcb96c249ef2.

{{</citation>}}


## cs.HC (3)



### (145/172) Understanding How to Inform Blind and Low-Vision Users about Data Privacy through Privacy Question Answering Assistants (Yuanyuan Feng et al., 2023)

{{<citation>}}

Yuanyuan Feng, Abhilasha Ravichander, Yaxing Yao, Shikun Zhang, Rex Chen, Shomir Wilson, Norman Sadeh. (2023)  
**Understanding How to Inform Blind and Low-Vision Users about Data Privacy through Privacy Question Answering Assistants**  

---
Primary Category: cs.HC  
Categories: cs-CR, cs-HC, cs.HC  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2310.08687v1)  

---


**ABSTRACT**  
Understanding and managing data privacy in the digital world can be challenging for sighted users, let alone blind and low-vision (BLV) users. There is limited research on how BLV users, who have special accessibility needs, navigate data privacy, and how potential privacy tools could assist them. We conducted an in-depth qualitative study with 21 US BLV participants to understand their data privacy risk perception and mitigation, as well as their information behaviors related to data privacy. We also explored BLV users' attitudes towards potential privacy question answering (Q&A) assistants that enable them to better navigate data privacy information. We found that BLV users face heightened security and privacy risks, but their risk mitigation is often insufficient. They do not necessarily seek data privacy information but clearly recognize the benefits of a potential privacy Q&A assistant. They also expect privacy Q&A assistants to possess cross-platform compatibility, support multi-modality, and demonstrate robust functionality. Our study sheds light on BLV users' expectations when it comes to usability, accessibility, trust and equity issues regarding digital data privacy.

{{</citation>}}


### (146/172) Jigsaw: Supporting Designers in Prototyping Multimodal Applications by Assembling AI Foundation Models (David Chuan-En Lin et al., 2023)

{{<citation>}}

David Chuan-En Lin, Nikolas Martelaro. (2023)  
**Jigsaw: Supporting Designers in Prototyping Multimodal Applications by Assembling AI Foundation Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08574v1)  

---


**ABSTRACT**  
Recent advancements in AI foundation models have made it possible for them to be utilized off-the-shelf for creative tasks, including ideating design concepts or generating visual prototypes. However, integrating these models into the creative process can be challenging as they often exist as standalone applications tailored to specific tasks. To address this challenge, we introduce Jigsaw, a prototype system that employs puzzle pieces as metaphors to represent foundation models. Jigsaw allows designers to combine different foundation model capabilities across various modalities by assembling compatible puzzle pieces. To inform the design of Jigsaw, we interviewed ten designers and distilled design goals. In a user study, we showed that Jigsaw enhanced designers' understanding of available foundation model capabilities, provided guidance on combining capabilities across different modalities and tasks, and served as a canvas to support design exploration, prototyping, and documentation.

{{</citation>}}


### (147/172) Receive, Reason, and React: Drive as You Say with Large Language Models in Autonomous Vehicles (Can Cui et al., 2023)

{{<citation>}}

Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Ziran Wang. (2023)  
**Receive, Reason, and React: Drive as You Say with Large Language Models in Autonomous Vehicles**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08034v1)  

---


**ABSTRACT**  
The fusion of human-centric design and artificial intelligence (AI) capabilities has opened up new possibilities for next-generation autonomous vehicles that go beyond transportation. These vehicles can dynamically interact with passengers and adapt to their preferences. This paper proposes a novel framework that leverages Large Language Models (LLMs) to enhance the decision-making process in autonomous vehicles. By utilizing LLMs' linguistic and contextual understanding abilities with specialized tools, we aim to integrate the language and reasoning capabilities of LLMs into autonomous vehicles. Our research includes experiments in HighwayEnv, a collection of environments for autonomous driving and tactical decision-making tasks, to explore LLMs' interpretation, interaction, and reasoning in various scenarios. We also examine real-time personalization, demonstrating how LLMs can influence driving behaviors based on verbal commands. Our empirical results highlight the substantial advantages of utilizing chain-of-thought prompting, leading to improved driving decisions, and showing the potential for LLMs to enhance personalized driving experiences through ongoing verbal feedback. The proposed framework aims to transform autonomous vehicle operations, offering personalized support, transparent decision-making, and continuous learning to enhance safety and effectiveness. We achieve user-centric, transparent, and adaptive autonomous driving ecosystems supported by the integration of LLMs into autonomous vehicles.

{{</citation>}}


## econ.EM (1)



### (148/172) Machine Learning Who to Nudge: Causal vs Predictive Targeting in a Field Experiment on Student Financial Aid Renewal (Susan Athey et al., 2023)

{{<citation>}}

Susan Athey, Niall Keleher, Jann Spiess. (2023)  
**Machine Learning Who to Nudge: Causal vs Predictive Targeting in a Field Experiment on Student Financial Aid Renewal**  

---
Primary Category: econ.EM  
Categories: cs-LG, econ-EM, econ.EM, stat-ME, stat-ML  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.08672v1)  

---


**ABSTRACT**  
In many settings, interventions may be more effective for some individuals than others, so that targeting interventions may be beneficial. We analyze the value of targeting in the context of a large-scale field experiment with over 53,000 college students, where the goal was to use "nudges" to encourage students to renew their financial-aid applications before a non-binding deadline. We begin with baseline approaches to targeting. First, we target based on a causal forest that estimates heterogeneous treatment effects and then assigns students to treatment according to those estimated to have the highest treatment effects. Next, we evaluate two alternative targeting policies, one targeting students with low predicted probability of renewing financial aid in the absence of the treatment, the other targeting those with high probability. The predicted baseline outcome is not the ideal criterion for targeting, nor is it a priori clear whether to prioritize low, high, or intermediate predicted probability. Nonetheless, targeting on low baseline outcomes is common in practice, for example because the relationship between individual characteristics and treatment effects is often difficult or impossible to estimate with historical data. We propose hybrid approaches that incorporate the strengths of both predictive approaches (accurate estimation) and causal approaches (correct criterion); we show that targeting intermediate baseline outcomes is most effective, while targeting based on low baseline outcomes is detrimental. In one year of the experiment, nudging all students improved early filing by an average of 6.4 percentage points over a baseline average of 37% filing, and we estimate that targeting half of the students using our preferred policy attains around 75% of this benefit.

{{</citation>}}


## cs.GR (1)



### (149/172) Discovering Fatigued Movements for Virtual Character Animation (Noshaba Cheema et al., 2023)

{{<citation>}}

Noshaba Cheema, Rui Xu, Nam Hee Kim, Perttu Hämäläinen, Vladislav Golyanik, Marc Habermann, Christian Theobalt, Philipp Slusallek. (2023)  
**Discovering Fatigued Movements for Virtual Character Animation**  

---
Primary Category: cs.GR  
Categories: I-3-7, cs-GR, cs-RO, cs.GR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08583v1)  

---


**ABSTRACT**  
Virtual character animation and movement synthesis have advanced rapidly during recent years, especially through a combination of extensive motion capture datasets and machine learning. A remaining challenge is interactively simulating characters that fatigue when performing extended motions, which is indispensable for the realism of generated animations. However, capturing such movements is problematic, as performing movements like backflips with fatigued variations up to exhaustion raises capture cost and risk of injury. Surprisingly, little research has been done on faithful fatigue modeling. To address this, we propose a deep reinforcement learning-based approach, which -- for the first time in literature -- generates control policies for full-body physically simulated agents aware of cumulative fatigue. For this, we first leverage Generative Adversarial Imitation Learning (GAIL) to learn an expert policy for the skill; Second, we learn a fatigue policy by limiting the generated constant torque bounds based on endurance time to non-linear, state- and time-dependent limits in the joint-actuation space using a Three-Compartment Controller (3CC) model. Our results demonstrate that agents can adapt to different fatigue and rest rates interactively, and discover realistic recovery strategies without the need for any captured data of fatigued movement.

{{</citation>}}


## cs.NI (2)



### (150/172) NetDiffusion: Network Data Augmentation Through Protocol-Constrained Traffic Generation (Xi Jiang et al., 2023)

{{<citation>}}

Xi Jiang, Shinan Liu, Aaron Gember-Jacobson, Arjun Nitin Bhagoji, Paul Schmitt, Francesco Bronzino, Nick Feamster. (2023)  
**NetDiffusion: Network Data Augmentation Through Protocol-Constrained Traffic Generation**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.08543v1)  

---


**ABSTRACT**  
Datasets of labeled network traces are essential for a multitude of machine learning (ML) tasks in networking, yet their availability is hindered by privacy and maintenance concerns, such as data staleness. To overcome this limitation, synthetic network traces can often augment existing datasets. Unfortunately, current synthetic trace generation methods, which typically produce only aggregated flow statistics or a few selected packet attributes, do not always suffice, especially when model training relies on having features that are only available from packet traces. This shortfall manifests in both insufficient statistical resemblance to real traces and suboptimal performance on ML tasks when employed for data augmentation. In this paper, we apply diffusion models to generate high-resolution synthetic network traffic traces. We present NetDiffusion, a tool that uses a finely-tuned, controlled variant of a Stable Diffusion model to generate synthetic network traffic that is high fidelity and conforms to protocol specifications. Our evaluation demonstrates that packet captures generated from NetDiffusion can achieve higher statistical similarity to real data and improved ML model performance than current state-of-the-art approaches (e.g., GAN-based approaches). Furthermore, our synthetic traces are compatible with common network analysis tools and support a myriad of network tasks, suggesting that NetDiffusion can serve a broader spectrum of network analysis and testing tasks, extending beyond ML-centric applications.

{{</citation>}}


### (151/172) ZEST: Attention-based Zero-Shot Learning for Unseen IoT Device Classification (Binghui Wu et al., 2023)

{{<citation>}}

Binghui Wu, Philipp Gysel, Dinil Mon Divakaran, Mohan Gurusamy. (2023)  
**ZEST: Attention-based Zero-Shot Learning for Unseen IoT Device Classification**  

---
Primary Category: cs.NI  
Categories: cs-CR, cs-LG, cs-NI, cs.NI  
Keywords: Attention, LSTM, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.08036v1)  

---


**ABSTRACT**  
Recent research works have proposed machine learning models for classifying IoT devices connected to a network. However, there is still a practical challenge of not having all devices (and hence their traffic) available during the training of a model. This essentially means, during the operational phase, we need to classify new devices not seen during the training phase. To address this challenge, we propose ZEST -- a ZSL (zero-shot learning) framework based on self-attention for classifying both seen and unseen devices. ZEST consists of i) a self-attention based network feature extractor, termed SANE, for extracting latent space representations of IoT traffic, ii) a generative model that trains a decoder using latent features to generate pseudo data, and iii) a supervised model that is trained on the generated pseudo data for classifying devices. We carry out extensive experiments on real IoT traffic data; our experiments demonstrate i) ZEST achieves significant improvement (in terms of accuracy) over the baselines; ii) ZEST is able to better extract meaningful representations than LSTM which has been commonly used for modeling network traffic.

{{</citation>}}


## cs.CY (1)



### (152/172) Metrics for popularity bias in dynamic recommender systems (Valentijn Braun et al., 2023)

{{<citation>}}

Valentijn Braun, Debarati Bhaumik, Diptish Dey. (2023)  
**Metrics for popularity bias in dynamic recommender systems**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.08455v1)  

---


**ABSTRACT**  
Albeit the widespread application of recommender systems (RecSys) in our daily lives, rather limited research has been done on quantifying unfairness and biases present in such systems. Prior work largely focuses on determining whether a RecSys is discriminating or not but does not compute the amount of bias present in these systems. Biased recommendations may lead to decisions that can potentially have adverse effects on individuals, sensitive user groups, and society. Hence, it is important to quantify these biases for fair and safe commercial applications of these systems. This paper focuses on quantifying popularity bias that stems directly from the output of RecSys models, leading to over recommendation of popular items that are likely to be misaligned with user preferences. Four metrics to quantify popularity bias in RescSys over time in dynamic setting across different sensitive user groups have been proposed. These metrics have been demonstrated for four collaborative filtering based RecSys algorithms trained on two commonly used benchmark datasets in the literature. Results obtained show that the metrics proposed provide a comprehensive understanding of growing disparities in treatment between sensitive groups over time when used conjointly.

{{</citation>}}


## physics.comp-ph (1)



### (153/172) TensorMD: Scalable Tensor-Diagram based Machine Learning Interatomic Potential on Heterogeneous Many-Core Processors (Xin Chen et al., 2023)

{{<citation>}}

Xin Chen, Yucheng Ouyang, Xin Chen, Zhenchuan Chen, Rongfen Lin, Xingyu Gao, Lifang Wang, Fang Li, Yin Liu, Honghui Shang, Haifeng Song. (2023)  
**TensorMD: Scalable Tensor-Diagram based Machine Learning Interatomic Potential on Heterogeneous Many-Core Processors**  

---
Primary Category: physics.comp-ph  
Categories: cs-DC, physics-comp-ph, physics.comp-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08439v2)  

---


**ABSTRACT**  
Molecular dynamics simulations have emerged as a potent tool for investigating the physical properties and kinetic behaviors of materials at the atomic scale, particularly in extreme conditions. Ab initio accuracy is now achievable with machine learning based interatomic potentials. With recent advancements in high-performance computing, highly accurate and large-scale simulations become feasible. This study introduces TensorMD, a new machine learning interatomic potential (MLIP) model that integrates physical principles and tensor diagrams. The tensor formalism provides a more efficient computation and greater flexibility for use with other scientific codes. Additionally, we proposed several portable optimization strategies and developed a highly optimized version for the new Sunway supercomputer. Our optimized TensorMD can achieve unprecedented performance on the new Sunway, enabling simulations of up to 52 billion atoms with a time-to-solution of 31 ps/step/atom, setting new records for HPC + AI + MD.

{{</citation>}}


## cs.DC (4)



### (154/172) Cold Start Latency in Serverless Computing: A Systematic Review, Taxonomy, and Future Directions (Muhammed Golec et al., 2023)

{{<citation>}}

Muhammed Golec, Guneet Kaur Walia, Mohit Kumar, Felix Cuadrado, Sukhpal Singh Gill, Steve Uhlig. (2023)  
**Cold Start Latency in Serverless Computing: A Systematic Review, Taxonomy, and Future Directions**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08437v1)  

---


**ABSTRACT**  
Recently, academics and the corporate sector have paid attention to serverless computing, which enables dynamic scalability and an economic model. In serverless computing, users pay only for the time they actually spend using the resources. Although zero scaling optimises cost and resource utilisation, it is the fundamental reason for the serverless cold start problem. Various academic and corporate sector studies are being conducted to tackle the cold start problem, which has large research challenges. To study the "cold start" problem in serverless computing, this article provides a comprehensive literature overview of recent research. In addition, we present a detailed taxonomy of several approaches to addressing the issue of cold start latency in serverless computing. Several academic and industrial organisations have proposed methods for cutting down the cold start time and cold start frequency, and this taxonomy is being used to explore these methods. There are several categories in which a current study on cold start latency is organised: caching and application-level optimization-based solutions, as well as AI/ML-based solutions. We have analysed the current methods and grouped them into categories based on their commonalities and features. Finally, we conclude with a review of current challenges and possible future research directions.

{{</citation>}}


### (155/172) Performance/power assessment of CNN packages on embedded automotive platforms (Paolo Burgio et al., 2023)

{{<citation>}}

Paolo Burgio, Gianluca Brilli. (2023)  
**Performance/power assessment of CNN packages on embedded automotive platforms**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs-PF, cs.DC  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2310.08401v1)  

---


**ABSTRACT**  
The rise of power-efficient embedded computers based on highly-parallel accelerators opens a number of opportunities and challenges for researchers and engineers, and paved the way to the era of edge computing. At the same time, advances in embedded AI for object detection and categorization such as YOLO, GoogleNet and AlexNet reached an unprecedented level of accuracy (mean-Average Precision - mAP) and performance (Frames-Per-Second - FPS). Today, edge computers based on heterogeneous many-core systems are a predominant choice to deploy such systems in industry 4.0, wearable devices, and - our focus - autonomous driving systems. In these latter systems, engineers struggle to make reduced automotive power and size budgets co-exist with the accuracy and performance targets requested by autonomous driving. We aim at validating the effectiveness and efficiency of most recent networks on state-of-the-art platforms with embedded commercial-off-the-shelf System-on-Chips, such as Xavier AGX, Tegra X2 and Nano for NVIDIA and XCZU9EG and XCZU3EG of the Zynq UltraScale+ family, for the Xilinx counterpart. Our work aims at supporting engineers in choosing the most appropriate CNN package and computing system for their designs, and deriving guidelines for adequately sizing their systems.

{{</citation>}}


### (156/172) Leveraging DevOps for Scientific Computing (Paul Nuyujukian, 2023)

{{<citation>}}

Paul Nuyujukian. (2023)  
**Leveraging DevOps for Scientific Computing**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.08247v1)  

---


**ABSTRACT**  
Critical goals of scientific computing are to increase scientific rigor, reproducibility, and transparency while keeping up with ever-increasing computational demands. This work presents an integrated framework well-suited for data processing and analysis spanning individual, on-premises, and cloud environments. This framework leverages three well-established DevOps tools: 1) Git repositories linked to 2) CI/CD engines operating on 3) containers. It supports the full life-cycle of scientific data workflows with minimal friction between stages--including solutions for researchers who generate data. This is achieved by leveraging a single container that supports local, interactive user sessions and deployment in HPC or Kubernetes clusters. Combined with Git repositories integrated with CI/CD, this approach enables decentralized data pipelines across multiple, arbitrary computational environments. This framework has been successfully deployed and validated within our research group, spanning experimental acquisition systems and computational clusters with open-source, purpose-built GitLab CI/CD executors for slurm and Google Kubernetes Engine Autopilot. Taken together, this framework can increase the rigor, reproducibility, and transparency of compute-dependent scientific research.

{{</citation>}}


### (157/172) Boosting Client Selection of Federated Learning under Device and Data Heterogeneity (Shuaijun Chen et al., 2023)

{{<citation>}}

Shuaijun Chen, Omid Tavallaie, Michael Henri Hambali, Seid Miad Zandavi, Hamed Haddadi, Song Guo, Albert Y. Zomaya. (2023)  
**Boosting Client Selection of Federated Learning under Device and Data Heterogeneity**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AWS, Amazon  
[Paper Link](http://arxiv.org/abs/2310.08147v1)  

---


**ABSTRACT**  
Federated learning (FL) is a promising distributed learning framework designed for privacy-aware applications of resource-constrained devices. Without sharing data, FL trains a model on each device locally and builds the global model on the server by aggregating the trained models. To reduce the communication overhead, only a portion of client devices participate in each round of training. Random selection is the most common way of selecting client devices for training data in a round of FL. However, random client selection uses distributed data and computational resources inefficiently, as it does not take into account the hardware specifications and data distribution among clients. This paper proposes FedGRA, an adaptive fair client selection algorithm designed for FL applications with unbalanced, non-Identically and Independently Distributed (IID) data running on client devices with heterogeneous computing resources. FedGRA dynamically adjusts the set of selected clients at each round of training based on clients' trained models and their available computational resources. To find an optimal solution, we model the client selection problem of FL as a multi-objective optimization by using Grey Relational Analysis (GRA) theory. To examine the performance of our proposed method, we implement our contribution on Amazon Web Services (AWS) by using 50 Elastic Compute Cloud (EC2) instances with 4 different hardware configurations. The evaluation results reveal that our contribution improves convergence significantly and reduces the average client's waiting time compared to state-of-the-art methods.

{{</citation>}}


## eess.SY (2)



### (158/172) Introducing a Deep Neural Network-based Model Predictive Control Framework for Rapid Controller Implementation (David C. Gordon et al., 2023)

{{<citation>}}

David C. Gordon, Alexander Winkler, Julian Bedei, Patrick Schaber, Jakob Andert, Charles R. Koch. (2023)  
**Introducing a Deep Neural Network-based Model Predictive Control Framework for Rapid Controller Implementation**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.08392v1)  

---


**ABSTRACT**  
Model Predictive Control (MPC) provides an optimal control solution based on a cost function while allowing for the implementation of process constraints. As a model-based optimal control technique, the performance of MPC strongly depends on the model used where a trade-off between model computation time and prediction performance exists. One solution is the integration of MPC with a machine learning (ML) based process model which are quick to evaluate online. This work presents the experimental implementation of a deep neural network (DNN) based nonlinear MPC for Homogeneous Charge Compression Ignition (HCCI) combustion control. The DNN model consists of a Long Short-Term Memory (LSTM) network surrounded by fully connected layers which was trained using experimental engine data and showed acceptable prediction performance with under 5% error for all outputs. Using this model, the MPC is designed to track the Indicated Mean Effective Pressure (IMEP) and combustion phasing trajectories, while minimizing several parameters. Using the acados software package to enable the real-time implementation of the MPC on an ARM Cortex A72, the optimization calculations are completed within 1.4 ms. The external A72 processor is integrated with the prototyping engine controller using a UDP connection allowing for rapid experimental deployment of the NMPC. The IMEP trajectory following of the developed controller was excellent, with a root-mean-square error of 0.133 bar, in addition to observing process constraints.

{{</citation>}}


### (159/172) CLExtract: Recovering Highly Corrupted DVB/GSE Satellite Stream with Contrastive Learning (Minghao Lin et al., 2023)

{{<citation>}}

Minghao Lin, Minghao Cheng, Dongsheng Luo, Yueqi Chen. (2023)  
**CLExtract: Recovering Highly Corrupted DVB/GSE Satellite Stream with Contrastive Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.08210v1)  

---


**ABSTRACT**  
Since satellite systems are playing an increasingly important role in our civilization, their security and privacy weaknesses are more and more concerned. For example, prior work demonstrates that the communication channel between maritime VSAT and ground segment can be eavesdropped on using consumer-grade equipment. The stream decoder GSExtract developed in this prior work performs well for most packets but shows incapacity for corrupted streams. We discovered that such stream corruption commonly exists in not only Europe and North Atlantic areas but also Asian areas. In our experiment, using GSExtract, we are only able to decode 2.1\% satellite streams we eavesdropped on in Asia.   Therefore, in this work, we propose to use a contrastive learning technique with data augmentation to decode and recover such highly corrupted streams. Rather than rely on critical information in corrupted streams to search for headers and perform decoding, contrastive learning directly learns the features of packet headers at different protocol layers and identifies them in a stream sequence. By filtering them out, we can extract the innermost data payload for further analysis. Our evaluation shows that this new approach can successfully recover 71-99\% eavesdropped data hundreds of times faster speed than GSExtract. Besides, the effectiveness of our approach is not largely damaged when stream corruption becomes more severe.

{{</citation>}}


## stat.ML (2)



### (160/172) How Many Pretraining Tasks Are Needed for In-Context Learning of Linear Regression? (Jingfeng Wu et al., 2023)

{{<citation>}}

Jingfeng Wu, Difan Zou, Zixiang Chen, Vladimir Braverman, Quanquan Gu, Peter L. Bartlett. (2023)  
**How Many Pretraining Tasks Are Needed for In-Context Learning of Linear Regression?**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08391v1)  

---


**ABSTRACT**  
Transformers pretrained on diverse tasks exhibit remarkable in-context learning (ICL) capabilities, enabling them to solve unseen tasks solely based on input contexts without adjusting model parameters. In this paper, we study ICL in one of its simplest setups: pretraining a linearly parameterized single-layer linear attention model for linear regression with a Gaussian prior. We establish a statistical task complexity bound for the attention model pretraining, showing that effective pretraining only requires a small number of independent tasks. Furthermore, we prove that the pretrained model closely matches the Bayes optimal algorithm, i.e., optimally tuned ridge regression, by achieving nearly Bayes optimal risk on unseen tasks under a fixed context length. These theoretical findings complement prior experimental research and shed light on the statistical foundations of ICL.

{{</citation>}}


### (161/172) Impact of multi-armed bandit strategies on deep recurrent reinforcement learning (Valentina Zangirolami et al., 2023)

{{<citation>}}

Valentina Zangirolami, Matteo Borrotti. (2023)  
**Impact of multi-armed bandit strategies on deep recurrent reinforcement learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08331v1)  

---


**ABSTRACT**  
Incomplete knowledge of the environment leads an agent to make decisions under uncertainty. One of the major dilemmas in Reinforcement Learning (RL) where an autonomous agent has to balance two contrasting needs in making its decisions is: exploiting the current knowledge of the environment to maximize the cumulative reward as well as exploring actions that allow improving the knowledge of the environment, hopefully leading to higher reward values (exploration-exploitation trade-off). Concurrently, another relevant issue regards the full observability of the states, which may not be assumed in all applications. Such as when only 2D images are considered as input in a RL approach used for finding the optimal action within a 3D simulation environment. In this work, we address these issues by deploying and testing several techniques to balance exploration and exploitation trade-off on partially observable systems for predicting steering wheels in autonomous driving scenario. More precisely, the final aim is to investigate the effects of using both stochastic and deterministic multi-armed bandit strategies coupled with a Deep Recurrent Q-Network. Additionally, we adapted and evaluated the impact of an innovative method to improve the learning phase of the underlying Convolutional Recurrent Neural Network. We aim to show that adaptive stochastic methods for exploration better approximate the trade-off between exploration and exploitation as, in general, Softmax and Max-Boltzmann strategies are able to outperform epsilon-greedy techniques.

{{</citation>}}


## eess.AS (2)



### (162/172) A cry for help: Early detection of brain injury in newborns (Charles C. Onu et al., 2023)

{{<citation>}}

Charles C. Onu, Samantha Latremouille, Arsenii Gorin, Junhao Wang, Uchenna Ekwochi, Peter O. Ubuane, Omolara A. Kehinde, Muhammad A. Salisu, Datonye Briggs, Yoshua Bengio, Doina Precup. (2023)  
**A cry for help: Early detection of brain injury in newborns**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS, q-bio-NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08338v2)  

---


**ABSTRACT**  
Since the 1960s, neonatal clinicians have known that newborns suffering from certain neurological conditions exhibit altered crying patterns such as the high-pitched cry in birth asphyxia. Despite an annual burden of over 1.5 million infant deaths and disabilities, early detection of neonatal brain injuries due to asphyxia remains a challenge, particularly in developing countries where the majority of births are not attended by a trained physician. Here we report on the first inter-continental clinical study to demonstrate that neonatal brain injury can be reliably determined from recorded infant cries using an AI algorithm we call Roseline. Previous and recent work has been limited by the lack of a large, high-quality clinical database of cry recordings, constraining the application of state-of-the-art machine learning. We develop a new training methodology for audio-based pathology detection models and evaluate this system on a large database of newborn cry sounds acquired from geographically diverse settings -- 5 hospitals across 3 continents. Our system extracts interpretable acoustic biomarkers that support clinical decisions and is able to accurately detect neurological injury from newborns' cries with an AUC of 92.5% (88.7% sensitivity at 80% specificity). Cry-based neurological monitoring opens the door for low-cost, easy-to-use, non-invasive and contact-free screening of at-risk babies, especially when integrated into simple devices like smartphones or neonatal ICU monitors. This would provide a reliable tool where there are no alternatives, but also curtail the need to regularly exert newborns to physically-exhausting or radiation-exposing assessments such as brain CT scans. This work sets the stage for embracing the infant cry as a vital sign and indicates the potential of AI-driven sound monitoring for the future of affordable healthcare.

{{</citation>}}


### (163/172) Fast Word Error Rate Estimation Using Self-Supervised Representations For Speech And Text (Chanho Park et al., 2023)

{{<citation>}}

Chanho Park, Chengsong Lu, Mingjie Chen, Thomas Hain. (2023)  
**Fast Word Error Rate Estimation Using Self-Supervised Representations For Speech And Text**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.08225v1)  

---


**ABSTRACT**  
The quality of automatic speech recognition (ASR) is typically measured by word error rate (WER). WER estimation is a task aiming to predict the WER of an ASR system, given a speech utterance and a transcription. This task has gained increasing attention while advanced ASR systems are trained on large amounts of data. In this case, WER estimation becomes necessary in many scenarios, for example, selecting training data with unknown transcription quality or estimating the testing performance of an ASR system without ground truth transcriptions. Facing large amounts of data, the computation efficiency of a WER estimator becomes essential in practical applications. However, previous works usually did not consider it as a priority. In this paper, a Fast WER estimator (Fe-WER) using self-supervised learning representation (SSLR) is introduced. The estimator is built upon SSLR aggregated by average pooling. The results show that Fe-WER outperformed the e-WER3 baseline relatively by 19.69% and 7.16% on Ted-Lium3 in both evaluation metrics of root mean square error and Pearson correlation coefficient, respectively. Moreover, the estimation weighted by duration was 10.43% when the target was 10.88%. Lastly, the inference speed was about 4x in terms of a real-time factor.

{{</citation>}}


## cs.CR (2)



### (164/172) 2SFGL: A Simple And Robust Protocol For Graph-Based Fraud Detection (Zhirui Pan et al., 2023)

{{<citation>}}

Zhirui Pan, Guangzhong Wang, Zhaoning Li, Lifeng Chen, Yang Bian, Zhongyuan Lai. (2023)  
**2SFGL: A Simple And Robust Protocol For Graph-Based Fraud Detection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Amazon, Financial, Fraud Detection, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2310.08335v1)  

---


**ABSTRACT**  
Financial crime detection using graph learning improves financial safety and efficiency. However, criminals may commit financial crimes across different institutions to avoid detection, which increases the difficulty of detection for financial institutions which use local data for graph learning. As most financial institutions are subject to strict regulations in regards to data privacy protection, the training data is often isolated and conventional learning technology cannot handle the problem. Federated learning (FL) allows multiple institutions to train a model without revealing their datasets to each other, hence ensuring data privacy protection. In this paper, we proposes a novel two-stage approach to federated graph learning (2SFGL): The first stage of 2SFGL involves the virtual fusion of multiparty graphs, and the second involves model training and inference on the virtual graph. We evaluate our framework on a conventional fraud detection task based on the FraudAmazonDataset and FraudYelpDataset. Experimental results show that integrating and applying a GCN (Graph Convolutional Network) with our 2SFGL framework to the same task results in a 17.6\%-30.2\% increase in performance on several typical metrics compared to the case only using FedAvg, while integrating GraphSAGE with 2SFGL results in a 6\%-16.2\% increase in performance compared to the case only using FedAvg. We conclude that our proposed framework is a robust and simple protocol which can be simply integrated to pre-existing graph-based fraud detection methods.

{{</citation>}}


### (165/172) Invisible Threats: Backdoor Attack in OCR Systems (Mauro Conti et al., 2023)

{{<citation>}}

Mauro Conti, Nicola Farronato, Stefanos Koffas, Luca Pajola, Stjepan Picek. (2023)  
**Invisible Threats: Backdoor Attack in OCR Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: NLP, OCR  
[Paper Link](http://arxiv.org/abs/2310.08259v1)  

---


**ABSTRACT**  
Optical Character Recognition (OCR) is a widely used tool to extract text from scanned documents. Today, the state-of-the-art is achieved by exploiting deep neural networks. However, the cost of this performance is paid at the price of system vulnerability. For instance, in backdoor attacks, attackers compromise the training phase by inserting a backdoor in the victim's model that will be activated at testing time by specific patterns while leaving the overall model performance intact. This work proposes a backdoor attack for OCR resulting in the injection of non-readable characters from malicious input images. This simple but effective attack exposes the state-of-the-art OCR weakness, making the extracted text correct to human eyes but simultaneously unusable for the NLP application that uses OCR as a preprocessing step. Experimental results show that the attacked models successfully output non-readable characters for around 90% of the poisoned instances without harming their performance for the remaining instances.

{{</citation>}}


## cs.IR (1)



### (166/172) Fine-Tuning LLaMA for Multi-Stage Text Retrieval (Xueguang Ma et al., 2023)

{{<citation>}}

Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, Jimmy Lin. (2023)  
**Fine-Tuning LLaMA for Multi-Stage Text Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2310.08319v1)  

---


**ABSTRACT**  
The effectiveness of multi-stage text retrieval has been solidly demonstrated since before the era of pre-trained language models. However, most existing studies utilize models that predate recent advances in large language models (LLMs). This study seeks to explore potential improvements that state-of-the-art LLMs can bring. We conduct a comprehensive study, fine-tuning the latest LLaMA model both as a dense retriever (RepLLaMA) and as a pointwise reranker (RankLLaMA) for both passage retrieval and document retrieval using the MS MARCO datasets. Our findings demonstrate that the effectiveness of large language models indeed surpasses that of smaller models. Additionally, since LLMs can inherently handle longer contexts, they can represent entire documents holistically, obviating the need for traditional segmenting and pooling strategies. Furthermore, evaluations on BEIR demonstrate that our RepLLaMA-RankLLaMA pipeline exhibits strong zero-shot effectiveness. Model checkpoints from this study are available on HuggingFace.

{{</citation>}}


## eess.SP (2)



### (167/172) Concealed Electronic Countermeasures of Radar Signal with Adversarial Examples (Ruinan Ma et al., 2023)

{{<citation>}}

Ruinan Ma, Canjie Zhu, Mingfeng Lu, Yunjie Li, Yu-an Tan, Ruibin Zhang, Ran Tao. (2023)  
**Concealed Electronic Countermeasures of Radar Signal with Adversarial Examples**  

---
Primary Category: eess.SP  
Categories: cs-AI, eess-SP, eess.SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08292v1)  

---


**ABSTRACT**  
Electronic countermeasures involving radar signals are an important aspect of modern warfare. Traditional electronic countermeasures techniques typically add large-scale interference signals to ensure interference effects, which can lead to attacks being too obvious. In recent years, AI-based attack methods have emerged that can effectively solve this problem, but the attack scenarios are currently limited to time domain radar signal classification. In this paper, we focus on the time-frequency images classification scenario of radar signals. We first propose an attack pipeline under the time-frequency images scenario and DITIMI-FGSM attack algorithm with high transferability. Then, we propose STFT-based time domain signal attack(STDS) algorithm to solve the problem of non-invertibility in time-frequency analysis, thus obtaining the time-domain representation of the interference signal. A large number of experiments show that our attack pipeline is feasible and the proposed attack method has a high success rate.

{{</citation>}}


### (168/172) A Carbon Tracking Model for Federated Learning: Impact of Quantization and Sparsification (Luca Barbieri et al., 2023)

{{<citation>}}

Luca Barbieri, Stefano Savazzi, Sanaz Kianoush, Monica Nicoli, Luigi Serio. (2023)  
**A Carbon Tracking Model for Federated Learning: Impact of Quantization and Sparsification**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: AI, Quantization  
[Paper Link](http://arxiv.org/abs/2310.08087v1)  

---


**ABSTRACT**  
Federated Learning (FL) methods adopt efficient communication technologies to distribute machine learning tasks across edge devices, reducing the overhead in terms of data storage and computational complexity compared to centralized solutions. Rather than moving large data volumes from producers (sensors, machines) to energy-hungry data centers, raising environmental concerns due to resource demands, FL provides an alternative solution to mitigate the energy demands of several learning tasks while enabling new Artificial Intelligence of Things (AIoT) applications. This paper proposes a framework for real-time monitoring of the energy and carbon footprint impacts of FL systems. The carbon tracking tool is evaluated for consensus (fully decentralized) and classical FL policies. For the first time, we present a quantitative evaluation of different computationally and communication efficient FL methods from the perspectives of energy consumption and carbon equivalent emissions, suggesting also general guidelines for energy-efficient design. Results indicate that consensus-driven FL implementations should be preferred for limiting carbon emissions when the energy efficiency of the communication is low (i.e., < 25 Kbit/Joule). Besides, quantization and sparsification operations are shown to strike a balance between learning performances and energy consumption, leading to sustainable FL designs.

{{</citation>}}


## eess.IV (1)



### (169/172) COVID-19 Detection Using Swin Transformer Approach from Computed Tomography Images (Kenan Morani, 2023)

{{<citation>}}

Kenan Morani. (2023)  
**COVID-19 Detection Using Swin Transformer Approach from Computed Tomography Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.08165v1)  

---


**ABSTRACT**  
The accurate and efficient diagnosis of COVID-19 is of paramount importance, particularly in the context of large-scale medical imaging datasets. In this preprint paper, we propose a novel approach for COVID-19 diagnosis using CT images that leverages the power of Swin Transformer models, state-of-the-art solutions in computer vision tasks. Our method includes a systematic approach for patient-level predictions, where individual CT slices are classified as COVID-19 or non-COVID, and the patient's overall diagnosis is determined through majority voting. The application of the Swin Transformer in this context results in patient-level predictions that demonstrate exceptional diagnostic accuracy. In terms of evaluation metrics, our approach consistently outperforms the baseline, as well as numerous competing methods, showcasing its effectiveness in COVID-19 diagnosis. The macro F1 score achieved by our model exceeds the baseline and offers a robust solution for accurate diagnosis.

{{</citation>}}


## cs.DS (1)



### (170/172) Core-sets for Fair and Diverse Data Summarization (Sepideh Mahabadi et al., 2023)

{{<citation>}}

Sepideh Mahabadi, Stojan Trajanovski. (2023)  
**Core-sets for Fair and Diverse Data Summarization**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-LG, cs.DS  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.08122v1)  

---


**ABSTRACT**  
We study core-set construction algorithms for the task of Diversity Maximization under fairness/partition constraint. Given a set of points $P$ in a metric space partitioned into $m$ groups, and given $k_1,\ldots,k_m$, the goal of this problem is to pick $k_i$ points from each group $i$ such that the overall diversity of the $k=\sum_i k_i$ picked points is maximized. We consider two natural diversity measures: sum-of-pairwise distances and sum-of-nearest-neighbor distances, and show improved core-set construction algorithms with respect to these measures. More precisely, we show the first constant factor core-set w.r.t. sum-of-pairwise distances whose size is independent of the size of the dataset and the aspect ratio. Second, we show the first core-set w.r.t. the sum-of-nearest-neighbor distances. Finally, we run several experiments showing the effectiveness of our core-set approach. In particular, we apply constrained diversity maximization to summarize a set of timed messages that takes into account the messages' recency. Specifically, the summary should include more recent messages compared to older ones. This is a real task in one of the largest communication platforms, affecting the experience of hundreds of millions daily active users. By utilizing our core-set method for this task, we achieve a 100x speed-up while losing the diversity by only a few percent. Moreover, our approach allows us to improve the space usage of the algorithm in the streaming setting.

{{</citation>}}


## cs.GT (1)



### (171/172) The Search-and-Mix Paradigm in Approximate Nash Equilibrium Algorithms (Xiaotie Deng et al., 2023)

{{<citation>}}

Xiaotie Deng, Dongchen Li, Hanyu Li. (2023)  
**The Search-and-Mix Paradigm in Approximate Nash Equilibrium Algorithms**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-DS, cs-GT, cs-LO, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08066v1)  

---


**ABSTRACT**  
AI in Math deals with mathematics in a constructive manner so that reasoning becomes automated, less laborious, and less error-prone. For algorithms, the question becomes how to automate analyses for specific problems. For the first time, this work provides an automatic method for approximation analysis on a well-studied problem in theoretical computer science: computing approximate Nash equilibria in two-player games. We observe that such algorithms can be reformulated into a search-and-mix paradigm, which involves a search phase followed by a mixing phase. By doing so, we are able to fully automate the procedure of designing and analyzing the mixing phase. For example, we illustrate how to perform our method with a program to analyze the approximation bounds of all the algorithms in the literature. Same approximation bounds are computed without any hand-written proof. Our automatic method heavily relies on the LP-relaxation structure in approximate Nash equilibria. Since many approximation algorithms and online algorithms adopt the LP relaxation, our approach may be extended to automate the analysis of other algorithms.

{{</citation>}}


## q-bio.BM (1)



### (172/172) ETDock: A Novel Equivariant Transformer for Protein-Ligand Docking (Yiqiang Yi et al., 2023)

{{<citation>}}

Yiqiang Yi, Xu Wan, Yatao Bian, Le Ou-Yang, Peilin Zhao. (2023)  
**ETDock: A Novel Equivariant Transformer for Protein-Ligand Docking**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.08061v1)  

---


**ABSTRACT**  
Predicting the docking between proteins and ligands is a crucial and challenging task for drug discovery. However, traditional docking methods mainly rely on scoring functions, and deep learning-based docking approaches usually neglect the 3D spatial information of proteins and ligands, as well as the graph-level features of ligands, which limits their performance. To address these limitations, we propose an equivariant transformer neural network for protein-ligand docking pose prediction. Our approach involves the fusion of ligand graph-level features by feature processing, followed by the learning of ligand and protein representations using our proposed TAMformer module. Additionally, we employ an iterative optimization approach based on the predicted distance matrix to generate refined ligand poses. The experimental results on real datasets show that our model can achieve state-of-the-art performance.

{{</citation>}}
