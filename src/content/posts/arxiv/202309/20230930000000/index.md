---
draft: false
title: "arXiv @ 2023.09.30"
date: 2023-09-30
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.30"
    identifier: arxiv_20230930
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (14)](#csro-14)
- [cs.LG (26)](#cslg-26)
- [cs.CV (39)](#cscv-39)
- [cs.NI (2)](#csni-2)
- [cs.CL (35)](#cscl-35)
- [cs.CR (2)](#cscr-2)
- [eess.IV (2)](#eessiv-2)
- [stat.ML (2)](#statml-2)
- [cs.HC (3)](#cshc-3)
- [cs.AI (7)](#csai-7)
- [cs.SD (4)](#cssd-4)
- [cs.DC (1)](#csdc-1)
- [cs.SI (1)](#cssi-1)
- [q-fin.GN (1)](#q-fingn-1)
- [physics.comp-ph (1)](#physicscomp-ph-1)
- [astro-ph.SR (1)](#astro-phsr-1)
- [q-bio.BM (1)](#q-biobm-1)
- [cs.CE (1)](#csce-1)
- [cs.DB (2)](#csdb-2)
- [cs.GT (1)](#csgt-1)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [eess.SY (1)](#eesssy-1)
- [cs.SE (2)](#csse-2)
- [eess.AS (1)](#eessas-1)

## cs.RO (14)



### (1/151) A Sign Language Recognition System with Pepper, Lightweight-Transformer, and LLM (JongYoon Lim et al., 2023)

{{<citation>}}

JongYoon Lim, Inkyu Sa, Bruce MacDonald, Ho Seok Ahn. (2023)  
**A Sign Language Recognition System with Pepper, Lightweight-Transformer, and LLM**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-CV, cs-HC, cs-RO, cs.RO  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16898v1)  

---


**ABSTRACT**  
This research explores using lightweight deep neural network architectures to enable the humanoid robot Pepper to understand American Sign Language (ASL) and facilitate non-verbal human-robot interaction. First, we introduce a lightweight and efficient model for ASL understanding optimized for embedded systems, ensuring rapid sign recognition while conserving computational resources. Building upon this, we employ large language models (LLMs) for intelligent robot interactions. Through intricate prompt engineering, we tailor interactions to allow the Pepper Robot to generate natural Co-Speech Gesture responses, laying the foundation for more organic and intuitive humanoid-robot dialogues. Finally, we present an integrated software pipeline, embodying advancements in a socially aware AI interaction model. Leveraging the Pepper Robot's capabilities, we demonstrate the practicality and effectiveness of our approach in real-world scenarios. The results highlight a profound potential for enhancing human-robot interaction through non-verbal interactions, bridging communication gaps, and making technology more accessible and understandable.

{{</citation>}}


### (2/151) An MCTS-DRL Based Obstacle and Occlusion Avoidance Methodology in Robotic Follow-Ahead Applications (Sahar Leisiazar et al., 2023)

{{<citation>}}

Sahar Leisiazar, Edward J. Park, Angelica Lim, Mo Chen. (2023)  
**An MCTS-DRL Based Obstacle and Occlusion Avoidance Methodology in Robotic Follow-Ahead Applications**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16884v1)  

---


**ABSTRACT**  
We propose a novel methodology for robotic follow-ahead applications that address the critical challenge of obstacle and occlusion avoidance. Our approach effectively navigates the robot while ensuring avoidance of collisions and occlusions caused by surrounding objects. To achieve this, we developed a high-level decision-making algorithm that generates short-term navigational goals for the mobile robot. Monte Carlo Tree Search is integrated with a Deep Reinforcement Learning method to enhance the performance of the decision-making process and generate more reliable navigational goals. Through extensive experimentation and analysis, we demonstrate the effectiveness and superiority of our proposed approach in comparison to the existing follow-ahead human-following robotic methods. Our code is available at https://github.com/saharLeisiazar/follow-ahead-ros.

{{</citation>}}


### (3/151) Predicting Object Interactions with Behavior Primitives: An Application in Stowing Tasks (Haonan Chen et al., 2023)

{{<citation>}}

Haonan Chen, Yilong Niu, Kaiwen Hong, Shuijing Liu, Yixuan Wang, Yunzhu, Katherine Driggs-Campbell. (2023)  
**Predicting Object Interactions with Behavior Primitives: An Application in Stowing Tasks**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.16873v1)  

---


**ABSTRACT**  
Stowing, the task of placing objects in cluttered shelves or bins, is a common task in warehouse and manufacturing operations. However, this task is still predominantly carried out by human workers as stowing is challenging to automate due to the complex multi-object interactions and long-horizon nature of the task. Previous works typically involve extensive data collection and costly human labeling of semantic priors across diverse object categories. This paper presents a method to learn a generalizable robot stowing policy from predictive model of object interactions and a single demonstration with behavior primitives. We propose a novel framework that utilizes Graph Neural Networks to predict object interactions within the parameter space of behavioral primitives. We further employ primitive-augmented trajectory optimization to search the parameters of a predefined library of heterogeneous behavioral primitives to instantiate the control action. Our framework enables robots to proficiently execute long-horizon stowing tasks with a few keyframes (3-4) from a single demonstration. Despite being solely trained in a simulation, our framework demonstrates remarkable generalization capabilities. It efficiently adapts to a broad spectrum of real-world conditions, including various shelf widths, fluctuating quantities of objects, and objects with diverse attributes such as sizes and shapes.

{{</citation>}}


### (4/151) Social Navigation in Crowded Environments with Model Predictive Control and Deep Learning-Based Human Trajectory Prediction (Viet-Anh Le et al., 2023)

{{<citation>}}

Viet-Anh Le, Behdad Chalaki, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Jovin D'sa, Ehsan Moradi-Pari. (2023)  
**Social Navigation in Crowded Environments with Model Predictive Control and Deep Learning-Based Human Trajectory Prediction**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.16838v1)  

---


**ABSTRACT**  
Crowd navigation has received increasing attention from researchers over the last few decades, resulting in the emergence of numerous approaches aimed at addressing this problem to date. Our proposed approach couples agent motion prediction and planning to avoid the freezing robot problem while simultaneously capturing multi-agent social interactions by utilizing a state-of-the-art trajectory prediction model i.e., social long short-term memory model (Social-LSTM). Leveraging the output of Social-LSTM for the prediction of future trajectories of pedestrians at each time-step given the robot's possible actions, our framework computes the optimal control action using Model Predictive Control (MPC) for the robot to navigate among pedestrians. We demonstrate the effectiveness of our proposed approach in multiple scenarios of simulated crowd navigation and compare it against several state-of-the-art reinforcement learning-based methods.

{{</citation>}}


### (5/151) An Attentional Recurrent Neural Network for Occlusion-Aware Proactive Anomaly Detection in Field Robot Navigation (Andre Schreiber et al., 2023)

{{<citation>}}

Andre Schreiber, Tianchen Ji, D. Livingston McPherson, Katherine Driggs-Campbell. (2023)  
**An Attentional Recurrent Neural Network for Occlusion-Aware Proactive Anomaly Detection in Field Robot Navigation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2309.16826v1)  

---


**ABSTRACT**  
The use of mobile robots in unstructured environments like the agricultural field is becoming increasingly common. The ability for such field robots to proactively identify and avoid failures is thus crucial for ensuring efficiency and avoiding damage. However, the cluttered field environment introduces various sources of noise (such as sensor occlusions) that make proactive anomaly detection difficult. Existing approaches can show poor performance in sensor occlusion scenarios as they typically do not explicitly model occlusions and only leverage current sensory inputs. In this work, we present an attention-based recurrent neural network architecture for proactive anomaly detection that fuses current sensory inputs and planned control actions with a latent representation of prior robot state. We enhance our model with an explicitly-learned model of sensor occlusion that is used to modulate the use of our latent representation of prior robot state. Our method shows improved anomaly detection performance and enables mobile field robots to display increased resilience to predicting false positives regarding navigation failure during periods of sensor occlusion, particularly in cases where all sensors are briefly occluded. Our code is available at: https://github.com/andreschreiber/roar

{{</citation>}}


### (6/151) Semantic Scene Difference Detection in Daily Life Patroling by Mobile Robots using Pre-Trained Large-Scale Vision-Language Model (Yoshiki Obinata et al., 2023)

{{<citation>}}

Yoshiki Obinata, Kento Kawaharazuka, Naoaki Kanazawa, Naoya Yamaguchi, Naoto Tsukamoto, Iori Yanokura, Shingo Kitagawa, Koki Shinjo, Kei Okada, Masayuki Inaba. (2023)  
**Semantic Scene Difference Detection in Daily Life Patroling by Mobile Robots using Pre-Trained Large-Scale Vision-Language Model**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.16552v1)  

---


**ABSTRACT**  
It is important for daily life support robots to detect changes in their environment and perform tasks. In the field of anomaly detection in computer vision, probabilistic and deep learning methods have been used to calculate the image distance. These methods calculate distances by focusing on image pixels. In contrast, this study aims to detect semantic changes in the daily life environment using the current development of large-scale vision-language models. Using its Visual Question Answering (VQA) model, we propose a method to detect semantic changes by applying multiple questions to a reference image and a current image and obtaining answers in the form of sentences. Unlike deep learning-based methods in anomaly detection, this method does not require any training or fine-tuning, is not affected by noise, and is sensitive to semantic state changes in the real world. In our experiments, we demonstrated the effectiveness of this method by applying it to a patrol task in a real-life environment using a mobile robot, Fetch Mobile Manipulator. In the future, it may be possible to add explanatory power to changes in the daily life environment through spoken language.

{{</citation>}}


### (7/151) QwenGrasp: A Usage of Large Vision Language Model for Target-oriented Grasping (Xinyu Chen et al., 2023)

{{<citation>}}

Xinyu Chen, Jian Yang, Zonghan He, Haobin Yang, Qi Zhao, Yuhui Shi. (2023)  
**QwenGrasp: A Usage of Large Vision Language Model for Target-oriented Grasping**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16426v1)  

---


**ABSTRACT**  
The ability for robotic systems to understand human language and execute grasping actions is a pivotal challenge in the field of robotics. In target-oriented grasping, prior researches achieve matching human textual commands with images of target objects. However, these works are hard to understand complex or flexible instructions. Moreover, these works lack the capability to autonomously assess the feasibility of instructions, leading to blindly execute grasping tasks even there is no target object. In this paper, we introduce a combination model called QwenGrasp, which combines a large vision language model with a 6-DoF grasp network. By leveraging a pre-trained large vision language model, our approach is capable of working in open-world with natural human language environments, accepting complex and flexible instructions. Furthermore, the specialized grasp network ensures the effectiveness of the generated grasp pose. A series of experiments conducted in real world environment show that our method exhibits a superior ability to comprehend human intent. Additionally, when accepting erroneous instructions, our approach has the capability to suspend task execution and provide feedback to humans, improving safety.

{{</citation>}}


### (8/151) Intrinsic Language-Guided Exploration for Complex Long-Horizon Robotic Manipulation Tasks (Eleftherios Triantafyllidis et al., 2023)

{{<citation>}}

Eleftherios Triantafyllidis, Filippos Christianos, Zhibin Li. (2023)  
**Intrinsic Language-Guided Exploration for Complex Long-Horizon Robotic Manipulation Tasks**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16347v1)  

---


**ABSTRACT**  
Current reinforcement learning algorithms struggle in sparse and complex environments, most notably in long-horizon manipulation tasks entailing a plethora of different sequences. In this work, we propose the Intrinsically Guided Exploration from Large Language Models (IGE-LLMs) framework. By leveraging LLMs as an assistive intrinsic reward, IGE-LLMs guides the exploratory process in reinforcement learning to address intricate long-horizon with sparse rewards robotic manipulation tasks. We evaluate our framework and related intrinsic learning methods in an environment challenged with exploration, and a complex robotic manipulation task challenged by both exploration and long-horizons. Results show IGE-LLMs (i) exhibit notably higher performance over related intrinsic methods and the direct use of LLMs in decision-making, (ii) can be combined and complement existing learning methods highlighting its modularity, (iii) are fairly insensitive to different intrinsic scaling parameters, and (iv) maintain robustness against increased levels of uncertainty and horizons.

{{</citation>}}


### (9/151) DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models (Licheng Wen et al., 2023)

{{<citation>}}

Licheng Wen, Daocheng Fu, Xin Li, Xinyu Cai, Tao Ma, Pinlong Cai, Min Dou, Botian Shi, Liang He, Yu Qiao. (2023)  
**DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-RO, cs.RO  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.16292v1)  

---


**ABSTRACT**  
Recent advancements in autonomous driving have relied on data-driven approaches, which are widely adopted but face challenges including dataset bias, overfitting, and uninterpretability. Drawing inspiration from the knowledge-driven nature of human driving, we explore the question of how to instill similar capabilities into autonomous driving systems and summarize a paradigm that integrates an interactive environment, a driver agent, as well as a memory component to address this question. Leveraging large language models with emergent abilities, we propose the DiLu framework, which combines a Reasoning and a Reflection module to enable the system to perform decision-making based on common-sense knowledge and evolve continuously. Extensive experiments prove DiLu's capability to accumulate experience and demonstrate a significant advantage in generalization ability over reinforcement learning-based methods. Moreover, DiLu is able to directly acquire experiences from real-world datasets which highlights its potential to be deployed on practical autonomous driving systems. To the best of our knowledge, we are the first to instill knowledge-driven capability into autonomous driving systems from the perspective of how humans drive.

{{</citation>}}


### (10/151) Laboratory Automation: Precision Insertion with Adaptive Fingers utilizing Contact through Sliding with Tactile-based Pose Estimation (Sameer Pai et al., 2023)

{{<citation>}}

Sameer Pai, Kuniyuki Takahashi, Shimpei Masuda, Naoki Fukaya, Koki Yamane, Avinash Ummadisingu. (2023)  
**Laboratory Automation: Precision Insertion with Adaptive Fingers utilizing Contact through Sliding with Tactile-based Pose Estimation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2309.16170v1)  

---


**ABSTRACT**  
Micro well-plates are commonly used apparatus in chemical and biological experiments that are a few centimeters in thickness with wells in them. The task we aim to solve is to place (insert) them onto a well-plate holder with grooves a few millimeters in height. Our insertion task has the following facets: 1) There is uncertainty in the detection of the position and pose of the well-plate and well-plate holder, 2) the accuracy required is in the order of millimeter to sub-millimeter, 3) the well-plate holder is not fastened, and moves with external force, 4) the groove is shallow, and 5) the width of the groove is small. Addressing these challenges, we developed a) an adaptive finger gripper with accurate detection of finger position (for (1)), b) grasped object pose estimation using tactile sensors (for (1)), c) a method to insert the well-plate into the target holder by sliding the well-plate while maintaining contact with the edge of the holder (for (2-4)), and d) estimating the orientation of the edge and aligning the well-plate so that the holder does not move when maintaining contact with the edge (for (5)). We show a significantly high success rate on the insertion task of the well-plate, even though under added noise.   An accompanying video is available at the following link: https://drive.google.com/file/d/1UxyJ3XIxqXPnHcpfw-PYs5T5oYQxoc6i/view?usp=sharing

{{</citation>}}


### (11/151) Learning to Terminate in Object Navigation (Yuhang Song et al., 2023)

{{<citation>}}

Yuhang Song, Anh Nguyen, Chun-Yi Lee. (2023)  
**Learning to Terminate in Object Navigation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16164v1)  

---


**ABSTRACT**  
This paper tackles the critical challenge of object navigation in autonomous navigation systems, particularly focusing on the problem of target approach and episode termination in environments with long optimal episode length in Deep Reinforcement Learning (DRL) based methods. While effective in environment exploration and object localization, conventional DRL methods often struggle with optimal path planning and termination recognition due to a lack of depth information. To overcome these limitations, we propose a novel approach, namely the Depth-Inference Termination Agent (DITA), which incorporates a supervised model called the Judge Model to implicitly infer object-wise depth and decide termination jointly with reinforcement learning. We train our judge model along with reinforcement learning in parallel and supervise the former efficiently by reward signal. Our evaluation shows the method is demonstrating superior performance, we achieve a 9.3% gain on success rate than our baseline method across all room types and gain 51.2% improvements on long episodes environment while maintaining slightly better Success Weighted by Path Length (SPL). Code and resources, visualization are available at: https://github.com/HuskyKingdom/DITA_acml2023

{{</citation>}}


### (12/151) D$^3$Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation (Yixuan Wang et al., 2023)

{{<citation>}}

Yixuan Wang, Zhuoran Li, Mingtong Zhang, Katherine Driggs-Campbell, Jiajun Wu, Li Fei-Fei, Yunzhu Li. (2023)  
**D$^3$Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.16118v1)  

---


**ABSTRACT**  
Scene representation has been a crucial design choice in robotic manipulation systems. An ideal representation should be 3D, dynamic, and semantic to meet the demands of diverse manipulation tasks. However, previous works often lack all three properties simultaneously. In this work, we introduce D$^3$Fields - dynamic 3D descriptor fields. These fields capture the dynamics of the underlying 3D environment and encode both semantic features and instance masks. Specifically, we project arbitrary 3D points in the workspace onto multi-view 2D visual observations and interpolate features derived from foundational models. The resulting fused descriptor fields allow for flexible goal specifications using 2D images with varied contexts, styles, and instances. To evaluate the effectiveness of these descriptor fields, we apply our representation to a wide range of robotic manipulation tasks in a zero-shot manner. Through extensive evaluation in both real-world scenarios and simulations, we demonstrate that D$^3$Fields are both generalizable and effective for zero-shot robotic manipulation tasks. In quantitative comparisons with state-of-the-art dense descriptors, such as Dense Object Nets and DINO, D$^3$Fields exhibit significantly better generalization abilities and manipulation accuracy.

{{</citation>}}


### (13/151) Comparing Active Learning Performance Driven by Gaussian Processes or Bayesian Neural Networks for Constrained Trajectory Exploration (Sapphira Akins et al., 2023)

{{<citation>}}

Sapphira Akins, Frances Zhu. (2023)  
**Comparing Active Learning Performance Driven by Gaussian Processes or Bayesian Neural Networks for Constrained Trajectory Exploration**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.16114v1)  

---


**ABSTRACT**  
Robots with increasing autonomy progress our space exploration capabilities, particularly for in-situ exploration and sampling to stand in for human explorers. Currently, humans drive robots to meet scientific objectives, but depending on the robot's location, the exchange of information and driving commands between the human operator and robot may cause undue delays in mission fulfillment. An autonomous robot encoded with a scientific objective and an exploration strategy incurs no communication delays and can fulfill missions more quickly. Active learning algorithms offer this capability of intelligent exploration, but the underlying model structure varies the performance of the active learning algorithm in accurately forming an understanding of the environment. In this paper, we investigate the performance differences between active learning algorithms driven by Gaussian processes or Bayesian neural networks for exploration strategies encoded on agents that are constrained in their trajectories, like planetary surface rovers. These two active learning strategies were tested in a simulation environment against science-blind strategies to predict the spatial distribution of a variable of interest along multiple datasets. The performance metrics of interest are model accuracy in root mean squared (RMS) error, training time, model convergence, total distance traveled until convergence, and total samples until convergence. Active learning strategies encoded with Gaussian processes require less computation to train, converge to an accurate model more quickly, and propose trajectories of shorter distance, except in a few complex environments in which Bayesian neural networks achieve a more accurate model in the large data regime due to their more expressive functional bases. The paper concludes with advice on when and how to implement either exploration strategy for future space missions.

{{</citation>}}


### (14/151) Infer and Adapt: Bipedal Locomotion Reward Learning from Demonstrations via Inverse Reinforcement Learning (Feiyang Wu et al., 2023)

{{<citation>}}

Feiyang Wu, Zhaoyuan Gu, Hanran Wu, Anqi Wu, Ye Zhao. (2023)  
**Infer and Adapt: Bipedal Locomotion Reward Learning from Demonstrations via Inverse Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16074v1)  

---


**ABSTRACT**  
Enabling bipedal walking robots to learn how to maneuver over highly uneven, dynamically changing terrains is challenging due to the complexity of robot dynamics and interacted environments. Recent advancements in learning from demonstrations have shown promising results for robot learning in complex environments. While imitation learning of expert policies has been well-explored, the study of learning expert reward functions is largely under-explored in legged locomotion. This paper brings state-of-the-art Inverse Reinforcement Learning (IRL) techniques to solving bipedal locomotion problems over complex terrains. We propose algorithms for learning expert reward functions, and we subsequently analyze the learned functions. Through nonlinear function approximation, we uncover meaningful insights into the expert's locomotion strategies. Furthermore, we empirically demonstrate that training a bipedal locomotion policy with the inferred reward functions enhances its walking performance on unseen terrains, highlighting the adaptability offered by reward learning.

{{</citation>}}


## cs.LG (26)



### (15/151) Algorithmic Recourse for Anomaly Detection in Multivariate Time Series (Xiao Han et al., 2023)

{{<citation>}}

Xiao Han, Lu Zhang, Yongkai Wu, Shuhan Yuan. (2023)  
**Algorithmic Recourse for Anomaly Detection in Multivariate Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2309.16896v1)  

---


**ABSTRACT**  
Anomaly detection in multivariate time series has received extensive study due to the wide spectrum of applications. An anomaly in multivariate time series usually indicates a critical event, such as a system fault or an external attack. Therefore, besides being effective in anomaly detection, recommending anomaly mitigation actions is also important in practice yet under-investigated. In this work, we focus on algorithmic recourse in time series anomaly detection, which is to recommend fixing actions on abnormal time series with a minimum cost so that domain experts can understand how to fix the abnormal behavior. To this end, we propose an algorithmic recourse framework, called RecAD, which can recommend recourse actions to flip the abnormal time steps. Experiments on two synthetic and one real-world datasets show the effectiveness of our framework.

{{</citation>}}


### (16/151) Sourcing Investment Targets for Venture and Growth Capital Using Multivariate Time Series Transformer (Lele Cao et al., 2023)

{{<citation>}}

Lele Cao, Gustaf Halvardsson, Andrew McCornack, Vilhelm von Ehrenheim, Pawel Herman. (2023)  
**Sourcing Investment Targets for Venture and Growth Capital Using Multivariate Time Series Transformer**  

---
Primary Category: cs.LG  
Categories: 91B84 (Primary) 68T07 (Secondary), I-2-6; I-2-1; H-4-0, cs-AI, cs-CE, cs-LG, cs.LG, q-fin-PM  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16888v1)  

---


**ABSTRACT**  
This paper addresses the growing application of data-driven approaches within the Private Equity (PE) industry, particularly in sourcing investment targets (i.e., companies) for Venture Capital (VC) and Growth Capital (GC). We present a comprehensive review of the relevant approaches and propose a novel approach leveraging a Transformer-based Multivariate Time Series Classifier (TMTSC) for predicting the success likelihood of any candidate company. The objective of our research is to optimize sourcing performance for VC and GC investments by formally defining the sourcing problem as a multivariate time series classification task. We consecutively introduce the key components of our implementation which collectively contribute to the successful application of TMTSC in VC/GC sourcing: input features, model architecture, optimization target, and investor-centric data augmentation and split. Our extensive experiments on four datasets, benchmarked towards three popular baselines, demonstrate the effectiveness of our approach in improving decision making within the VC and GC industry.

{{</citation>}}


### (17/151) Message Propagation Through Time: An Algorithm for Sequence Dependency Retention in Time Series Modeling (Shaoming Xu et al., 2023)

{{<citation>}}

Shaoming Xu, Ankush Khandelwal, Arvind Renganathan, Vipin Kumar. (2023)  
**Message Propagation Through Time: An Algorithm for Sequence Dependency Retention in Time Series Modeling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.16882v1)  

---


**ABSTRACT**  
Time series modeling, a crucial area in science, often encounters challenges when training Machine Learning (ML) models like Recurrent Neural Networks (RNNs) using the conventional mini-batch training strategy that assumes independent and identically distributed (IID) samples and initializes RNNs with zero hidden states. The IID assumption ignores temporal dependencies among samples, resulting in poor performance. This paper proposes the Message Propagation Through Time (MPTT) algorithm to effectively incorporate long temporal dependencies while preserving faster training times relative to the stateful solutions. MPTT utilizes two memory modules to asynchronously manage initial hidden states for RNNs, fostering seamless information exchange between samples and allowing diverse mini-batches throughout epochs. MPTT further implements three policies to filter outdated and preserve essential information in the hidden states to generate informative initial hidden states for RNNs, facilitating robust training. Experimental results demonstrate that MPTT outperforms seven strategies on four climate datasets with varying levels of temporal dependencies.

{{</citation>}}


### (18/151) FENDA-FL: Personalized Federated Learning on Heterogeneous Clinical Datasets (Fatemeh Tavakoli et al., 2023)

{{<citation>}}

Fatemeh Tavakoli, D. B. Emerson, John Jewell, Amrit Krishnan, Yuchong Zhang, Amol Verma, Fahad Razak. (2023)  
**FENDA-FL: Personalized Federated Learning on Heterogeneous Clinical Datasets**  

---
Primary Category: cs.LG  
Categories: 68T07, cs-LG, cs.LG  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2309.16825v1)  

---


**ABSTRACT**  
Federated learning (FL) is increasingly being recognized as a key approach to overcoming the data silos that so frequently obstruct the training and deployment of machine-learning models in clinical settings. This work contributes to a growing body of FL research specifically focused on clinical applications along three important directions. First, an extension of the FENDA method (Kim et al., 2016) to the FL setting is proposed. Experiments conducted on the FLamby benchmarks (du Terrail et al., 2022a) and GEMINI datasets (Verma et al., 2017) show that the approach is robust to heterogeneous clinical data and often outperforms existing global and personalized FL techniques. Further, the experimental results represent substantive improvements over the original FLamby benchmarks and expand such benchmarks to include evaluation of personalized FL methods. Finally, we advocate for a comprehensive checkpointing and evaluation framework for FL to better reflect practical settings and provide multiple baselines for comparison.

{{</citation>}}


### (19/151) PROSE: Predicting Operators and Symbolic Expressions using Multimodal Transformers (Yuxuan Liu et al., 2023)

{{<citation>}}

Yuxuan Liu, Zecheng Zhang, Hayden Schaeffer. (2023)  
**PROSE: Predicting Operators and Symbolic Expressions using Multimodal Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16816v1)  

---


**ABSTRACT**  
Approximating nonlinear differential equations using a neural network provides a robust and efficient tool for various scientific computing tasks, including real-time predictions, inverse problems, optimal controls, and surrogate modeling. Previous works have focused on embedding dynamical systems into networks through two approaches: learning a single solution operator (i.e., the mapping from input parametrized functions to solutions) or learning the governing system of equations (i.e., the constitutive model relative to the state variables). Both of these approaches yield different representations for the same underlying data or function. Additionally, observing that families of differential equations often share key characteristics, we seek one network representation across a wide range of equations. Our method, called Predicting Operators and Symbolic Expressions (PROSE), learns maps from multimodal inputs to multimodal outputs, capable of generating both numerical predictions and mathematical equations. By using a transformer structure and a feature fusion approach, our network can simultaneously embed sets of solution operators for various parametric differential equations using a single trained network. Detailed experiments demonstrate that the network benefits from its multimodal nature, resulting in improved prediction accuracy and better generalization. The network is shown to be able to handle noise in the data and errors in the symbolic representation, including noisy numerical values, model misspecification, and erroneous addition or deletion of terms. PROSE provides a new neural network framework for differential equations which allows for more flexibility and generality in learning operators and governing equations from data.

{{</citation>}}


### (20/151) De-SaTE: Denoising Self-attention Transformer Encoders for Li-ion Battery Health Prognostics (Gaurav Shinde et al., 2023)

{{<citation>}}

Gaurav Shinde, Rohan Mohapatra, Pooja Krishan, Saptarshi Sengupta. (2023)  
**De-SaTE: Denoising Self-attention Transformer Encoders for Li-ion Battery Health Prognostics**  

---
Primary Category: cs.LG  
Categories: I-2-0, cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.00023v1)  

---


**ABSTRACT**  
Lithium Ion (Li-ion) batteries have gained widespread popularity across various industries, from powering portable electronic devices to propelling electric vehicles and supporting energy storage systems. A central challenge in managing Li-ion batteries effectively is accurately predicting their Remaining Useful Life (RUL), which is a critical measure for proactive maintenance and predictive analytics. This study presents a novel approach that harnesses the power of multiple denoising modules, each trained to address specific types of noise commonly encountered in battery data. Specifically we use a denoising auto-encoder and a wavelet denoiser to generate encoded/decomposed representations, which are subsequently processed through dedicated self-attention transformer encoders. After extensive experimentation on the NASA and CALCE datasets, we are able to characterize a broad spectrum of health indicator estimations under a set of diverse noise patterns. We find that our reported error metrics on these datasets are on par or better with the best reported in recent literature.

{{</citation>}}


### (21/151) Neural scaling laws for phenotypic drug discovery (Drew Linsley et al., 2023)

{{<citation>}}

Drew Linsley, John Griffin, Jason Parker Brown, Adam N Roose, Michael Frank, Peter Linsley, Steven Finkbeiner, Jeremy Linsley. (2023)  
**Neural scaling laws for phenotypic drug discovery**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG, q-bio-QM  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.16773v1)  

---


**ABSTRACT**  
Recent breakthroughs by deep neural networks (DNNs) in natural language processing (NLP) and computer vision have been driven by a scale-up of models and data rather than the discovery of novel computing paradigms. Here, we investigate if scale can have a similar impact for models designed to aid small molecule drug discovery. We address this question through a large-scale and systematic analysis of how DNN size, data diet, and learning routines interact to impact accuracy on our Phenotypic Chemistry Arena (Pheno-CA) benchmark: a diverse set of drug development tasks posed on image-based high content screening data. Surprisingly, we find that DNNs explicitly supervised to solve tasks in the Pheno-CA do not continuously improve as their data and model size is scaled-up. To address this issue, we introduce a novel precursor task, the Inverse Biological Process (IBP), which is designed to resemble the causal objective functions that have proven successful for NLP. We indeed find that DNNs first trained with IBP then probed for performance on the Pheno-CA significantly outperform task-supervised DNNs. More importantly, the performance of these IBP-trained DNNs monotonically improves with data and model scale. Our findings reveal that the DNN ingredients needed to accurately solve small molecule drug development tasks are already in our hands, and project how much more experimental data is needed to achieve any desired level of improvement. We release our Pheno-CA benchmark and code to encourage further study of neural scaling laws for small molecule drug discovery.

{{</citation>}}


### (22/151) Discovering environments with XRM (Mohammad Pezeshki et al., 2023)

{{<citation>}}

Mohammad Pezeshki, Diane Bouchacourt, Mark Ibrahim, Nicolas Ballas, Pascal Vincent, David Lopez-Paz. (2023)  
**Discovering environments with XRM**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16748v1)  

---


**ABSTRACT**  
Successful out-of-distribution generalization requires environment annotations. Unfortunately, these are resource-intensive to obtain, and their relevance to model performance is limited by the expectations and perceptual biases of human annotators. Therefore, to enable robust AI systems across applications, we must develop algorithms to automatically discover environments inducing broad generalization. Current proposals, which divide examples based on their training error, suffer from one fundamental problem. These methods add hyper-parameters and early-stopping criteria that are impossible to tune without a validation set with human-annotated environments, the very information subject to discovery. In this paper, we propose Cross-Risk-Minimization (XRM) to address this issue. XRM trains two twin networks, each learning from one random half of the training data, while imitating confident held-out mistakes made by its sibling. XRM provides a recipe for hyper-parameter tuning, does not require early-stopping, and can discover environments for all training and validation data. Domain generalization algorithms built on top of XRM environments achieve oracle worst-group-accuracy, solving a long-standing problem in out-of-distribution generalization.

{{</citation>}}


### (23/151) Mixup Your Own Pairs (Yilei Wu et al., 2023)

{{<citation>}}

Yilei Wu, Zijian Dong, Chongyao Chen, Wangchunshu Zhou, Juan Helen Zhou. (2023)  
**Mixup Your Own Pairs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.16633v2)  

---


**ABSTRACT**  
In representation learning, regression has traditionally received less attention than classification. Directly applying representation learning techniques designed for classification to regression often results in fragmented representations in the latent space, yielding sub-optimal performance. In this paper, we argue that the potential of contrastive learning for regression has been overshadowed due to the neglect of two crucial aspects: ordinality-awareness and hardness. To address these challenges, we advocate "mixup your own contrastive pairs for supervised contrastive regression", instead of relying solely on real/augmented samples. Specifically, we propose Supervised Contrastive Learning for Regression with Mixup (SupReMix). It takes anchor-inclusive mixtures (mixup of the anchor and a distinct negative sample) as hard negative pairs and anchor-exclusive mixtures (mixup of two distinct negative samples) as hard positive pairs at the embedding level. This strategy formulates harder contrastive pairs by integrating richer ordinal information. Through extensive experiments on six regression datasets including 2D images, volumetric images, text, tabular data, and time-series signals, coupled with theoretical analysis, we demonstrate that SupReMix pre-training fosters continuous ordered representations of regression data, resulting in significant improvement in regression performance. Furthermore, SupReMix is superior to other approaches in a range of regression challenges including transfer learning, imbalanced training data, and scenarios with fewer training samples.

{{</citation>}}


### (24/151) Robust Offline Reinforcement Learning -- Certify the Confidence Interval (Jiarui Yao et al., 2023)

{{<citation>}}

Jiarui Yao, Simon Shaolei Du. (2023)  
**Robust Offline Reinforcement Learning -- Certify the Confidence Interval**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16631v2)  

---


**ABSTRACT**  
Currently, reinforcement learning (RL), especially deep RL, has received more and more attention in the research area. However, the security of RL has been an obvious problem due to the attack manners becoming mature. In order to defend against such adversarial attacks, several practical approaches are developed, such as adversarial training, data filtering, etc. However, these methods are mostly based on empirical algorithms and experiments, without rigorous theoretical analysis of the robustness of the algorithms. In this paper, we develop an algorithm to certify the robustness of a given policy offline with random smoothing, which could be proven and conducted as efficiently as ones without random smoothing. Experiments on different environments confirm the correctness of our algorithm.

{{</citation>}}


### (25/151) Can LLMs Effectively Leverage Graph Structural Information: When and Why (Jin Huang et al., 2023)

{{<citation>}}

Jin Huang, Xingjian Zhang, Qiaozhu Mei, Jiaqi Ma. (2023)  
**Can LLMs Effectively Leverage Graph Structural Information: When and Why**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16595v2)  

---


**ABSTRACT**  
This paper studies Large Language Models (LLMs) augmented with structured data--particularly graphs--a crucial data modality that remains underexplored in the LLM literature. We aim to understand when and why the incorporation of structural information inherent in graph data can improve the prediction performance of LLMs on node classification tasks with textual features. To address the ``when'' question, we examine a variety of prompting methods for encoding structural information, in settings where textual node features are either rich or scarce. For the ``why'' questions, we probe into two potential contributing factors to the LLM performance: data leakage and homophily. Our exploration of these questions reveals that (i) LLMs can benefit from structural information, especially when textual node features are scarce; (ii) there is no substantial evidence indicating that the performance of LLMs is significantly attributed to data leakage; and (iii) the performance of LLMs on a target node is strongly positively related to the local homophily ratio of the node\footnote{Codes and datasets are at: \url{https://github.com/TRAIS-Lab/LLM-Structured-Data}}.

{{</citation>}}


### (26/151) Augment to Interpret: Unsupervised and Inherently Interpretable Graph Embeddings (Gregory Scafarto et al., 2023)

{{<citation>}}

Gregory Scafarto, Madalina Ciortan, Simon Tihon, Quentin Ferre. (2023)  
**Augment to Interpret: Unsupervised and Inherently Interpretable Graph Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Embedding  
[Paper Link](http://arxiv.org/abs/2309.16564v1)  

---


**ABSTRACT**  
Unsupervised learning allows us to leverage unlabelled data, which has become abundantly available, and to create embeddings that are usable on a variety of downstream tasks. However, the typical lack of interpretability of unsupervised representation learning has become a limiting factor with regard to recent transparent-AI regulations. In this paper, we study graph representation learning and we show that data augmentation that preserves semantics can be learned and used to produce interpretations. Our framework, which we named INGENIOUS, creates inherently interpretable embeddings and eliminates the need for costly additional post-hoc analysis. We also introduce additional metrics addressing the lack of formalism and metrics in the understudied area of unsupervised-representation learning interpretability. Our results are supported by an experimental study applied to both graph-level and node-level tasks and show that interpretable embeddings provide state-of-the-art performance on subsequent downstream tasks.

{{</citation>}}


### (27/151) Uncertainty-Aware Decision Transformer for Stochastic Driving Environments (Zenan Li et al., 2023)

{{<citation>}}

Zenan Li, Fan Nie, Qiao Sun, Fang Da, Hang Zhao. (2023)  
**Uncertainty-Aware Decision Transformer for Stochastic Driving Environments**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16397v1)  

---


**ABSTRACT**  
Offline Reinforcement Learning (RL) has emerged as a promising framework for learning policies without active interactions, making it especially appealing for autonomous driving tasks. Recent successes of Transformers inspire casting offline RL as sequence modeling, which performs well in long-horizon tasks. However, they are overly optimistic in stochastic environments with incorrect assumptions that the same goal can be consistently achieved by identical actions. In this paper, we introduce an UNcertainty-awaRE deciSion Transformer (UNREST) for planning in stochastic driving environments without introducing additional transition or complex generative models. Specifically, UNREST estimates state uncertainties by the conditional mutual information between transitions and returns, and segments sequences accordingly. Discovering the `uncertainty accumulation' and `temporal locality' properties of driving environments, UNREST replaces the global returns in decision transformers with less uncertain truncated returns, to learn from true outcomes of agent actions rather than environment transitions. We also dynamically evaluate environmental uncertainty during inference for cautious planning. Extensive experimental results demonstrate UNREST's superior performance in various driving scenarios and the power of our uncertainty estimation strategy.

{{</citation>}}


### (28/151) MHG-GNN: Combination of Molecular Hypergraph Grammar with Graph Neural Network (Akihiro Kishimoto et al., 2023)

{{<citation>}}

Akihiro Kishimoto, Hiroshi Kajino, Masataka Hirose, Junta Fuchiwaki, Indra Priyadarsini, Lisa Hamada, Hajime Shinohara, Daiju Nakano, Seiji Takeda. (2023)  
**MHG-GNN: Combination of Molecular Hypergraph Grammar with Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.16374v1)  

---


**ABSTRACT**  
Property prediction plays an important role in material discovery. As an initial step to eventually develop a foundation model for material science, we introduce a new autoencoder called the MHG-GNN, which combines graph neural network (GNN) with Molecular Hypergraph Grammar (MHG). Results on a variety of property prediction tasks with diverse materials show that MHG-GNN is promising.

{{</citation>}}


### (29/151) Leveraging Pre-trained Language Models for Time Interval Prediction in Text-Enhanced Temporal Knowledge Graphs (Duygu Sezen Islakoglu et al., 2023)

{{<citation>}}

Duygu Sezen Islakoglu, Mel Chekol, Yannis Velegrakis. (2023)  
**Leveraging Pre-trained Language Models for Time Interval Prediction in Text-Enhanced Temporal Knowledge Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16357v1)  

---


**ABSTRACT**  
Most knowledge graph completion (KGC) methods learn latent representations of entities and relations of a given graph by mapping them into a vector space. Although the majority of these methods focus on static knowledge graphs, a large number of publicly available KGs contain temporal information stating the time instant/period over which a certain fact has been true. Such graphs are often known as temporal knowledge graphs. Furthermore, knowledge graphs may also contain textual descriptions of entities and relations. Both temporal information and textual descriptions are not taken into account during representation learning by static KGC methods, and only structural information of the graph is leveraged. Recently, some studies have used temporal information to improve link prediction, yet they do not exploit textual descriptions and do not support inductive inference (prediction on entities that have not been seen in training).   We propose a novel framework called TEMT that exploits the power of pre-trained language models (PLMs) for text-enhanced temporal knowledge graph completion. The knowledge stored in the parameters of a PLM allows TEMT to produce rich semantic representations of facts and to generalize on previously unseen entities. TEMT leverages textual and temporal information available in a KG, treats them separately, and fuses them to get plausibility scores of facts. Unlike previous approaches, TEMT effectively captures dependencies across different time points and enables predictions on unseen entities. To assess the performance of TEMT, we carried out several experiments including time interval prediction, both in transductive and inductive settings, and triple classification. The experimental results show that TEMT is competitive with the state-of-the-art.

{{</citation>}}


### (30/151) Transformer-VQ: Linear-Time Transformers via Vector Quantization (Lucas D. Lingle, 2023)

{{<citation>}}

Lucas D. Lingle. (2023)  
**Transformer-VQ: Linear-Time Transformers via Vector Quantization**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet, Quantization, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16354v1)  

---


**ABSTRACT**  
We introduce Transformer-VQ, a decoder-only transformer computing softmax-based dense self-attention in linear time. Transformer-VQ's efficient attention is enabled by vector-quantized keys and a novel caching mechanism. In large-scale experiments, Transformer-VQ is shown highly competitive in quality, with strong results on Enwik8 (0.99 bpb), PG-19 (26.6 ppl), and ImageNet64 (3.16 bpb). Code: https://github.com/transformer-vq/transformer_vq

{{</citation>}}


### (31/151) ShapeDBA: Generating Effective Time Series Prototypes using ShapeDTW Barycenter Averaging (Ali Ismail-Fawaz et al., 2023)

{{<citation>}}

Ali Ismail-Fawaz, Hassan Ismail Fawaz, François Petitjean, Maxime Devanne, Jonathan Weber, Stefano Berretti, Geoffrey I. Webb, Germain Forestier. (2023)  
**ShapeDBA: Generating Effective Time Series Prototypes using ShapeDTW Barycenter Averaging**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.16353v1)  

---


**ABSTRACT**  
Time series data can be found in almost every domain, ranging from the medical field to manufacturing and wireless communication. Generating realistic and useful exemplars and prototypes is a fundamental data analysis task. In this paper, we investigate a novel approach to generating realistic and useful exemplars and prototypes for time series data. Our approach uses a new form of time series average, the ShapeDTW Barycentric Average. We therefore turn our attention to accurately generating time series prototypes with a novel approach. The existing time series prototyping approaches rely on the Dynamic Time Warping (DTW) similarity measure such as DTW Barycentering Average (DBA) and SoftDBA. These last approaches suffer from a common problem of generating out-of-distribution artifacts in their prototypes. This is mostly caused by the DTW variant used and its incapability of detecting neighborhood similarities, instead it detects absolute similarities. Our proposed method, ShapeDBA, uses the ShapeDTW variant of DTW, that overcomes this issue. We chose time series clustering, a popular form of time series analysis to evaluate the outcome of ShapeDBA compared to the other prototyping approaches. Coupled with the k-means clustering algorithm, and evaluated on a total of 123 datasets from the UCR archive, our proposed averaging approach is able to achieve new state-of-the-art results in terms of Adjusted Rand Index.

{{</citation>}}


### (32/151) LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite (Artur P. Toshev et al., 2023)

{{<citation>}}

Artur P. Toshev, Gianluca Galletti, Fabian Fritz, Stefan Adami, Nikolaus A. Adams. (2023)  
**LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-flu-dyn  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.16342v1)  

---


**ABSTRACT**  
Machine learning has been successfully applied to grid-based PDE modeling in various scientific applications. However, learned PDE solvers based on Lagrangian particle discretizations, which are the preferred approach to problems with free surfaces or complex physics, remain largely unexplored. We present LagrangeBench, the first benchmarking suite for Lagrangian particle problems, focusing on temporal coarse-graining. In particular, our contribution is: (a) seven new fluid mechanics datasets (four in 2D and three in 3D) generated with the Smoothed Particle Hydrodynamics (SPH) method including the Taylor-Green vortex, lid-driven cavity, reverse Poiseuille flow, and dam break, each of which includes different physics like solid wall interactions or free surface, (b) efficient JAX-based API with various recent training strategies and neighbors search routine, and (c) JAX implementation of established Graph Neural Networks (GNNs) like GNS and SEGNN with baseline results. Finally, to measure the performance of learned surrogates we go beyond established position errors and introduce physical metrics like kinetic energy MSE and Sinkhorn distance for the particle distribution. Our codebase is available under the URL: https://github.com/tumaer/lagrangebench

{{</citation>}}


### (33/151) Efficiency Separation between RL Methods: Model-Free, Model-Based and Goal-Conditioned (Brieuc Pinon et al., 2023)

{{<citation>}}

Brieuc Pinon, Raphaël Jungers, Jean-Charles Delvenne. (2023)  
**Efficiency Separation between RL Methods: Model-Free, Model-Based and Goal-Conditioned**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16291v1)  

---


**ABSTRACT**  
We prove a fundamental limitation on the efficiency of a wide class of Reinforcement Learning (RL) algorithms. This limitation applies to model-free RL methods as well as a broad range of model-based methods, such as planning with tree search.   Under an abstract definition of this class, we provide a family of RL problems for which these methods suffer a lower bound exponential in the horizon for their interactions with the environment to find an optimal behavior. However, there exists a method, not tailored to this specific family of problems, which can efficiently solve the problems in the family.   In contrast, our limitation does not apply to several types of methods proposed in the literature, for instance, goal-conditioned methods or other algorithms that construct an inverse dynamics model.

{{</citation>}}


### (34/151) Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints (Chaoqi Wang et al., 2023)

{{<citation>}}

Chaoqi Wang, Yibo Jiang, Chenghao Yang, Han Liu, Yuxin Chen. (2023)  
**Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16240v1)  

---


**ABSTRACT**  
The increasing capabilities of large language models (LLMs) raise opportunities for artificial general intelligence but concurrently amplify safety concerns, such as potential misuse of AI systems, necessitating effective AI alignment. Reinforcement Learning from Human Feedback (RLHF) has emerged as a promising pathway towards AI alignment but brings forth challenges due to its complexity and dependence on a separate reward model. Direct Preference Optimization (DPO) has been proposed as an alternative, and it remains equivalent to RLHF under the reverse KL regularization constraint. This paper presents $f$-DPO, a generalized approach to DPO by incorporating diverse divergence constraints. We show that under certain $f$-divergences, including Jensen-Shannon divergence, forward KL divergences and $\alpha$-divergences, the complex relationship between the reward and optimal policy can also be simplified by addressing the Karush-Kuhn-Tucker conditions. This eliminates the need for estimating the normalizing constant in the Bradley-Terry model and enables a tractable mapping between the reward function and the optimal policy. Our approach optimizes LLMs to align with human preferences in a more efficient and supervised manner under a broad set of divergence constraints. Empirically, adopting these divergences ensures a balance between alignment performance and generation diversity. Importantly, $f$-DPO outperforms PPO-based methods in divergence efficiency, and divergence constraints directly influence expected calibration error (ECE).

{{</citation>}}


### (35/151) Multi-Modal Financial Time-Series Retrieval Through Latent Space Projections (Tom Bamford et al., 2023)

{{<citation>}}

Tom Bamford, Andrea Coletta, Elizabeth Fons, Sriram Gopalakrishnan, Svitlana Vyetrenko, Tucker Balch, Manuela Veloso. (2023)  
**Multi-Modal Financial Time-Series Retrieval Through Latent Space Projections**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-HC, cs-LG, cs.LG  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2309.16741v1)  

---


**ABSTRACT**  
Financial firms commonly process and store billions of time-series data, generated continuously and at a high frequency. To support efficient data storage and retrieval, specialized time-series databases and systems have emerged. These databases support indexing and querying of time-series by a constrained Structured Query Language(SQL)-like format to enable queries like "Stocks with monthly price returns greater than 5%", and expressed in rigid formats. However, such queries do not capture the intrinsic complexity of high dimensional time-series data, which can often be better described by images or language (e.g., "A stock in low volatility regime"). Moreover, the required storage, computational time, and retrieval complexity to search in the time-series space are often non-trivial. In this paper, we propose and demonstrate a framework to store multi-modal data for financial time-series in a lower-dimensional latent space using deep encoders, such that the latent space projections capture not only the time series trends but also other desirable information or properties of the financial time-series data (such as price volatility). Moreover, our approach allows user-friendly query interfaces, enabling natural language text or sketches of time-series, for which we have developed intuitive interfaces. We demonstrate the advantages of our method in terms of computational efficiency and accuracy on real historical data as well as synthetic data, and highlight the utility of latent-space projections in the storage and retrieval of financial time-series data with intuitive query modalities.

{{</citation>}}


### (36/151) Unmasking the Chameleons: A Benchmark for Out-of-Distribution Detection in Medical Tabular Data (Mohammad Azizmalayeri et al., 2023)

{{<citation>}}

Mohammad Azizmalayeri, Ameen Abu-Hanna, Giovanni Ciná. (2023)  
**Unmasking the Chameleons: A Benchmark for Out-of-Distribution Detection in Medical Tabular Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.16220v1)  

---


**ABSTRACT**  
Despite their success, Machine Learning (ML) models do not generalize effectively to data not originating from the training distribution. To reliably employ ML models in real-world healthcare systems and avoid inaccurate predictions on out-of-distribution (OOD) data, it is crucial to detect OOD samples. Numerous OOD detection approaches have been suggested in other fields - especially in computer vision - but it remains unclear whether the challenge is resolved when dealing with medical tabular data. To answer this pressing need, we propose an extensive reproducible benchmark to compare different methods across a suite of tests including both near and far OODs. Our benchmark leverages the latest versions of eICU and MIMIC-IV, two public datasets encompassing tens of thousands of ICU patients in several hospitals. We consider a wide array of density-based methods and SOTA post-hoc detectors across diverse predictive architectures, including MLP, ResNet, and Transformer. Our findings show that i) the problem appears to be solved for far-OODs, but remains open for near-OODs; ii) post-hoc methods alone perform poorly, but improve substantially when coupled with distance-based mechanisms; iii) the transformer architecture is far less overconfident compared to MLP and ResNet.

{{</citation>}}


### (37/151) Pushing Large Language Models to the 6G Edge: Vision, Challenges, and Opportunities (Zheng Lin et al., 2023)

{{<citation>}}

Zheng Lin, Guanqiao Qu, Qiyuan Chen, Xianhao Chen, Zhe Chen, Kaibin Huang. (2023)  
**Pushing Large Language Models to the 6G Edge: Vision, Challenges, and Opportunities**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16739v1)  

---


**ABSTRACT**  
Large language models (LLMs), which have shown remarkable capabilities, are revolutionizing AI development and potentially shaping our future. However, given their multimodality, the status quo cloud-based deployment faces some critical challenges: 1) long response time; 2) high bandwidth costs; and 3) the violation of data privacy. 6G mobile edge computing (MEC) systems may resolve these pressing issues. In this article, we explore the potential of deploying LLMs at the 6G edge. We start by introducing killer applications powered by multimodal LLMs, including robotics and healthcare, to highlight the need for deploying LLMs in the vicinity of end users. Then, we identify the critical challenges for LLM deployment at the edge and envision the 6G MEC architecture for LLMs. Furthermore, we delve into two design aspects, i.e., edge training and edge inference for LLMs. In both aspects, considering the inherent resource limitations at the edge, we discuss various cutting-edge techniques, including split learning/inference, parameter-efficient fine-tuning, quantization, and parameter-sharing inference, to facilitate the efficient deployment of LLMs. This article serves as a position paper for thoroughly identifying the motivation, challenges, and pathway for empowering LLMs at the 6G edge.

{{</citation>}}


### (38/151) Distill to Delete: Unlearning in Graph Networks with Knowledge Distillation (Yash Sinha et al., 2023)

{{<citation>}}

Yash Sinha, Murari Mandal, Mohan Kankanhalli. (2023)  
**Distill to Delete: Unlearning in Graph Networks with Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.16173v1)  

---


**ABSTRACT**  
Graph unlearning has emerged as a pivotal method to delete information from a pre-trained graph neural network (GNN). One may delete nodes, a class of nodes, edges, or a class of edges. An unlearning method enables the GNN model to comply with data protection regulations (i.e., the right to be forgotten), adapt to evolving data distributions, and reduce the GPU-hours carbon footprint by avoiding repetitive retraining. Existing partitioning and aggregation-based methods have limitations due to their poor handling of local graph dependencies and additional overhead costs. More recently, GNNDelete offered a model-agnostic approach that alleviates some of these issues. Our work takes a novel approach to address these challenges in graph unlearning through knowledge distillation, as it distills to delete in GNN (D2DGN). It is a model-agnostic distillation framework where the complete graph knowledge is divided and marked for retention and deletion. It performs distillation with response-based soft targets and feature-based node embedding while minimizing KL divergence. The unlearned model effectively removes the influence of deleted graph elements while preserving knowledge about the retained graph elements. D2DGN surpasses the performance of existing methods when evaluated on various real-world graph datasets by up to $43.1\%$ (AUC) in edge and node unlearning tasks. Other notable advantages include better efficiency, better performance in removing target elements, preservation of performance for the retained elements, and zero overhead costs. Notably, our D2DGN surpasses the state-of-the-art GNNDelete in AUC by $2.4\%$, improves membership inference ratio by $+1.3$, requires $10.2\times10^6$ fewer FLOPs per forward pass and up to $\mathbf{3.2}\times$ faster.

{{</citation>}}


### (39/151) Generative Semi-supervised Learning with Meta-Optimized Synthetic Samples (Shin'ya Yamaguchi, 2023)

{{<citation>}}

Shin'ya Yamaguchi. (2023)  
**Generative Semi-supervised Learning with Meta-Optimized Synthetic Samples**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.16143v1)  

---


**ABSTRACT**  
Semi-supervised learning (SSL) is a promising approach for training deep classification models using labeled and unlabeled datasets. However, existing SSL methods rely on a large unlabeled dataset, which may not always be available in many real-world applications due to legal constraints (e.g., GDPR). In this paper, we investigate the research question: Can we train SSL models without real unlabeled datasets? Instead of using real unlabeled datasets, we propose an SSL method using synthetic datasets generated from generative foundation models trained on datasets containing millions of samples in diverse domains (e.g., ImageNet). Our main concepts are identifying synthetic samples that emulate unlabeled samples from generative foundation models and training classifiers using these synthetic samples. To achieve this, our method is formulated as an alternating optimization problem: (i) meta-learning of generative foundation models and (ii) SSL of classifiers using real labeled and synthetic unlabeled samples. For (i), we propose a meta-learning objective that optimizes latent variables to generate samples that resemble real labeled samples and minimize the validation loss. For (ii), we propose a simple unsupervised loss function that regularizes the feature extractors of classifiers to maximize the performance improvement obtained from synthetic samples. We confirm that our method outperforms baselines using generative foundation models on SSL. We also demonstrate that our methods outperform SSL using real unlabeled datasets in scenarios with extremely small amounts of labeled datasets. This suggests that synthetic samples have the potential to provide improvement gains more efficiently than real unlabeled data.

{{</citation>}}


### (40/151) E2Net: Resource-Efficient Continual Learning with Elastic Expansion Network (RuiQi Liu et al., 2023)

{{<citation>}}

RuiQi Liu, Boyu Diao, Libo Huang, Zhulin An, Yongjun Xu. (2023)  
**E2Net: Resource-Efficient Continual Learning with Elastic Expansion Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Network Distillation  
[Paper Link](http://arxiv.org/abs/2309.16117v1)  

---


**ABSTRACT**  
Continual Learning methods are designed to learn new tasks without erasing previous knowledge. However, Continual Learning often requires massive computational power and storage capacity for satisfactory performance. In this paper, we propose a resource-efficient continual learning method called the Elastic Expansion Network (E2Net). Leveraging core subnet distillation and precise replay sample selection, E2Net achieves superior average accuracy and diminished forgetting within the same computational and storage constraints, all while minimizing processing time. In E2Net, we propose Representative Network Distillation to identify the representative core subnet by assessing parameter quantity and output similarity with the working network, distilling analogous subnets within the working network to mitigate reliance on rehearsal buffers and facilitating knowledge transfer across previous tasks. To enhance storage resource utilization, we then propose Subnet Constraint Experience Replay to optimize rehearsal efficiency through a sample storage strategy based on the structures of representative networks. Extensive experiments conducted predominantly on cloud environments with diverse datasets and also spanning the edge environment demonstrate that E2Net consistently outperforms state-of-the-art methods. In addition, our method outperforms competitors in terms of both storage and computational requirements.

{{</citation>}}


## cs.CV (39)



### (41/151) Superpixel Transformers for Efficient Semantic Segmentation (Alex Zihao Zhu et al., 2023)

{{<citation>}}

Alex Zihao Zhu, Jieru Mei, Siyuan Qiao, Hang Yan, Yukun Zhu, Liang-Chieh Chen, Henrik Kretzschmar. (2023)  
**Superpixel Transformers for Efficient Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning, Semantic Segmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16889v2)  

---


**ABSTRACT**  
Semantic segmentation, which aims to classify every pixel in an image, is a key task in machine perception, with many applications across robotics and autonomous driving. Due to the high dimensionality of this task, most existing approaches use local operations, such as convolutions, to generate per-pixel features. However, these methods are typically unable to effectively leverage global context information due to the high computational costs of operating on a dense image. In this work, we propose a solution to this issue by leveraging the idea of superpixels, an over-segmentation of the image, and applying them with a modern transformer framework. In particular, our model learns to decompose the pixel space into a spatially low dimensional superpixel space via a series of local cross-attentions. We then apply multi-head self-attention to the superpixels to enrich the superpixel features with global context and then directly produce a class prediction for each superpixel. Finally, we directly project the superpixel class predictions back into the pixel space using the associations between the superpixels and the image pixel features. Reasoning in the superpixel space allows our method to be substantially more computationally efficient compared to convolution-based decoder methods. Yet, our method achieves state-of-the-art performance in semantic segmentation due to the rich superpixel features generated by the global self-attention mechanism. Our experiments on Cityscapes and ADE20K demonstrate that our method matches the state of the art in terms of accuracy, while outperforming in terms of model parameters and latency.

{{</citation>}}


### (42/151) LEF: Late-to-Early Temporal Fusion for LiDAR 3D Object Detection (Tong He et al., 2023)

{{<citation>}}

Tong He, Pei Sun, Zhaoqi Leng, Chenxi Liu, Dragomir Anguelov, Mingxing Tan. (2023)  
**LEF: Late-to-Early Temporal Fusion for LiDAR 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.16870v1)  

---


**ABSTRACT**  
We propose a late-to-early recurrent feature fusion scheme for 3D object detection using temporal LiDAR point clouds. Our main motivation is fusing object-aware latent embeddings into the early stages of a 3D object detector. This feature fusion strategy enables the model to better capture the shapes and poses for challenging objects, compared with learning from raw points directly. Our method conducts late-to-early feature fusion in a recurrent manner. This is achieved by enforcing window-based attention blocks upon temporally calibrated and aligned sparse pillar tokens. Leveraging bird's eye view foreground pillar segmentation, we reduce the number of sparse history features that our model needs to fuse into its current frame by 10$\times$. We also propose a stochastic-length FrameDrop training technique, which generalizes the model to variable frame lengths at inference for improved performance without retraining. We evaluate our method on the widely adopted Waymo Open Dataset and demonstrate improvement on 3D object detection against the baseline model, especially for the challenging category of large objects.

{{</citation>}}


### (43/151) Sketch2CADScript: 3D Scene Reconstruction from 2D Sketch using Visual Transformer and Rhino Grasshopper (Hong-Bin Yang, 2023)

{{<citation>}}

Hong-Bin Yang. (2023)  
**Sketch2CADScript: 3D Scene Reconstruction from 2D Sketch using Visual Transformer and Rhino Grasshopper**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16850v1)  

---


**ABSTRACT**  
Existing 3D model reconstruction methods typically produce outputs in the form of voxels, point clouds, or meshes. However, each of these approaches has its limitations and may not be suitable for every scenario. For instance, the resulting model may exhibit a rough surface and distorted structure, making manual editing and post-processing challenging for humans. In this paper, we introduce a novel 3D reconstruction method designed to address these issues. We trained a visual transformer to predict a "scene descriptor" from a single wire-frame image. This descriptor encompasses crucial information, including object types and parameters such as position, rotation, and size. With the predicted parameters, a 3D scene can be reconstructed using 3D modeling software like Blender or Rhino Grasshopper which provides a programmable interface, resulting in finely and easily editable 3D models. To evaluate the proposed model, we created two datasets: one featuring simple scenes and another with complex scenes. The test results demonstrate the model's ability to accurately reconstruct simple scenes but reveal its challenges with more complex ones.

{{</citation>}}


### (44/151) Space-Time Attention with Shifted Non-Local Search (Kent Gauen et al., 2023)

{{<citation>}}

Kent Gauen, Stanley Chan. (2023)  
**Space-Time Attention with Shifted Non-Local Search**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.16849v1)  

---


**ABSTRACT**  
Efficiently computing attention maps for videos is challenging due to the motion of objects between frames. While a standard non-local search is high-quality for a window surrounding each query point, the window's small size cannot accommodate motion. Methods for long-range motion use an auxiliary network to predict the most similar key coordinates as offsets from each query location. However, accurately predicting this flow field of offsets remains challenging, even for large-scale networks. Small spatial inaccuracies significantly impact the attention module's quality. This paper proposes a search strategy that combines the quality of a non-local search with the range of predicted offsets. The method, named Shifted Non-Local Search, executes a small grid search surrounding the predicted offsets to correct small spatial errors. Our method's in-place computation consumes 10 times less memory and is over 3 times faster than previous work. Experimentally, correcting the small spatial errors improves the video frame alignment quality by over 3 dB PSNR. Our search upgrades existing space-time attention modules, which improves video denoising results by 0.30 dB PSNR for a 7.5% increase in overall runtime. We integrate our space-time attention module into a UNet-like architecture to achieve state-of-the-art results on video denoising.

{{</citation>}}


### (45/151) Ultra-low-power Image Classification on Neuromorphic Hardware (Gregor Lenz et al., 2023)

{{<citation>}}

Gregor Lenz, Garrick Orchard, Sadique Sheik. (2023)  
**Ultra-low-power Image Classification on Neuromorphic Hardware**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2309.16795v1)  

---


**ABSTRACT**  
Spiking neural networks (SNNs) promise ultra-low-power applications by exploiting temporal and spatial sparsity. The number of binary activations, called spikes, is proportional to the power consumed when executed on neuromorphic hardware. Training such SNNs using backpropagation through time for vision tasks that rely mainly on spatial features is computationally costly. Training a stateless artificial neural network (ANN) to then convert the weights to an SNN is a straightforward alternative when it comes to image recognition datasets. Most conversion methods rely on rate coding in the SNN to represent ANN activation, which uses enormous amounts of spikes and, therefore, energy to encode information. Recently, temporal conversion methods have shown promising results requiring significantly fewer spikes per neuron, but sometimes complex neuron models. We propose a temporal ANN-to-SNN conversion method, which we call Quartz, that is based on the time to first spike (TTFS). Quartz achieves high classification accuracy and can be easily implemented on neuromorphic hardware while using the least amount of synaptic operations and memory accesses. It incurs a cost of two additional synapses per neuron compared to previous temporal conversion methods, which are readily available on neuromorphic hardware. We benchmark Quartz on MNIST, CIFAR10, and ImageNet in simulation to show the benefits of our method and follow up with an implementation on Loihi, a neuromorphic chip by Intel. We provide evidence that temporal coding has advantages in terms of power consumption, throughput, and latency for similar classification accuracy. Our code and models are publicly available.

{{</citation>}}


### (46/151) Prompt-Enhanced Self-supervised Representation Learning for Remote Sensing Image Understanding (Mingming Zhang et al., 2023)

{{<citation>}}

Mingming Zhang, Qingjie Liu, Yunhong Wang. (2023)  
**Prompt-Enhanced Self-supervised Representation Learning for Remote Sensing Image Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.00022v1)  

---


**ABSTRACT**  
Learning representations through self-supervision on a large-scale, unlabeled dataset has proven to be highly effective for understanding diverse images, such as those used in remote sensing image analysis. However, remote sensing images often have complex and densely populated scenes, with multiple land objects and no clear foreground objects. This intrinsic property can lead to false positive pairs in contrastive learning, or missing contextual information in reconstructive learning, which can limit the effectiveness of existing self-supervised learning methods. To address these problems, we propose a prompt-enhanced self-supervised representation learning method that uses a simple yet efficient pre-training pipeline. Our approach involves utilizing original image patches as a reconstructive prompt template, and designing a prompt-enhanced generative branch that provides contextual information through semantic consistency constraints. We collected a dataset of over 1.28 million remote sensing images that is comparable to the popular ImageNet dataset, but without specific temporal or geographical constraints. Our experiments show that our method outperforms fully supervised learning models and state-of-the-art self-supervised learning methods on various downstream tasks, including land cover classification, semantic segmentation, object detection, and instance segmentation. These results demonstrate that our approach learns impressive remote sensing representations with high generalization and transferability.

{{</citation>}}


### (47/151) Learning to Transform for Generalizable Instance-wise Invariance (Utkarsh Singhal et al., 2023)

{{<citation>}}

Utkarsh Singhal, Carlos Esteves, Ameesh Makadia, Stella X. Yu. (2023)  
**Learning to Transform for Generalizable Instance-wise Invariance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.16672v1)  

---


**ABSTRACT**  
Computer vision research has long aimed to build systems that are robust to spatial transformations found in natural data. Traditionally, this is done using data augmentation or hard-coding invariances into the architecture. However, too much or too little invariance can hurt, and the correct amount is unknown a priori and dependent on the instance. Ideally, the appropriate invariance would be learned from data and inferred at test-time.   We treat invariance as a prediction problem. Given any image, we use a normalizing flow to predict a distribution over transformations and average the predictions over them. Since this distribution only depends on the instance, we can align instances before classifying them and generalize invariance across classes. The same distribution can also be used to adapt to out-of-distribution poses. This normalizing flow is trained end-to-end and can learn a much larger range of transformations than Augerino and InstaAug. When used as data augmentation, our method shows accuracy and robustness gains on CIFAR 10, CIFAR10-LT, and TinyImageNet.

{{</citation>}}


### (48/151) Demystifying CLIP Data (Hu Xu et al., 2023)

{{<citation>}}

Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, Christoph Feichtenhofer. (2023)  
**Demystifying CLIP Data**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.16671v3)  

---


**ABSTRACT**  
Contrastive Language-Image Pre-training (CLIP) is an approach that has advanced research and applications in computer vision, fueling modern recognition systems and generative models. We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective. However, CLIP only provides very limited information about its data and how it has been collected, leading to works that aim to reproduce CLIP's data by filtering with its model parameters. In this work, we intend to reveal CLIP's data curation approach and in our pursuit of making it open to the community introduce Metadata-Curated Language-Image Pre-training (MetaCLIP). MetaCLIP takes a raw data pool and metadata (derived from CLIP's concepts) and yields a balanced subset over the metadata distribution. Our experimental study rigorously isolates the model and training settings, concentrating solely on data. MetaCLIP applied to CommonCrawl with 400M image-text data pairs outperforms CLIP's data on multiple standard benchmarks. In zero-shot ImageNet classification, MetaCLIP achieves 70.8% accuracy, surpassing CLIP's 68.3% on ViT-B models. Scaling to 1B data, while maintaining the same training budget, attains 72.4%. Our observations hold across various model sizes, exemplified by ViT-H achieving 80.5%, without any bells-and-whistles. Curation code and training data distribution on metadata is made available at https://github.com/facebookresearch/MetaCLIP.

{{</citation>}}


### (49/151) SA2-Net: Scale-aware Attention Network for Microscopic Image Segmentation (Mustansar Fiaz et al., 2023)

{{<citation>}}

Mustansar Fiaz, Moein Heidari, Rao Muhammad Anwer, Hisham Cholakkal. (2023)  
**SA2-Net: Scale-aware Attention Network for Microscopic Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.16661v1)  

---


**ABSTRACT**  
Microscopic image segmentation is a challenging task, wherein the objective is to assign semantic labels to each pixel in a given microscopic image. While convolutional neural networks (CNNs) form the foundation of many existing frameworks, they often struggle to explicitly capture long-range dependencies. Although transformers were initially devised to address this issue using self-attention, it has been proven that both local and global features are crucial for addressing diverse challenges in microscopic images, including variations in shape, size, appearance, and target region density. In this paper, we introduce SA2-Net, an attention-guided method that leverages multi-scale feature learning to effectively handle diverse structures within microscopic images. Specifically, we propose scale-aware attention (SA2) module designed to capture inherent variations in scales and shapes of microscopic regions, such as cells, for accurate segmentation. This module incorporates local attention at each level of multi-stage features, as well as global attention across multiple resolutions. Furthermore, we address the issue of blurred region boundaries (e.g., cell boundaries) by introducing a novel upsampling strategy called the Adaptive Up-Attention (AuA) module. This module enhances the discriminative ability for improved localization of microscopic regions using an explicit attention mechanism. Extensive experiments on five challenging datasets demonstrate the benefits of our SA2-Net model. Our source code is publicly available at \url{https://github.com/mustansarfiaz/SA2-Net}.

{{</citation>}}


### (50/151) Visual In-Context Learning for Few-Shot Eczema Segmentation (Neelesh Kumar et al., 2023)

{{<citation>}}

Neelesh Kumar, Oya Aran, Venugopal Vasudevan. (2023)  
**Visual In-Context Learning for Few-Shot Eczema Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot, GPT  
[Paper Link](http://arxiv.org/abs/2309.16656v1)  

---


**ABSTRACT**  
Automated diagnosis of eczema from digital camera images is crucial for developing applications that allow patients to self-monitor their recovery. An important component of this is the segmentation of eczema region from such images. Current methods for eczema segmentation rely on deep neural networks such as convolutional (CNN)-based U-Net or transformer-based Swin U-Net. While effective, these methods require high volume of annotated data, which can be difficult to obtain. Here, we investigate the capabilities of visual in-context learning that can perform few-shot eczema segmentation with just a handful of examples and without any need for retraining models. Specifically, we propose a strategy for applying in-context learning for eczema segmentation with a generalist vision model called SegGPT. When benchmarked on a dataset of annotated eczema images, we show that SegGPT with just 2 representative example images from the training dataset performs better (mIoU: 36.69) than a CNN U-Net trained on 428 images (mIoU: 32.60). We also discover that using more number of examples for SegGPT may in fact be harmful to its performance. Our result highlights the importance of visual in-context learning in developing faster and better solutions to skin imaging tasks. Our result also paves the way for developing inclusive solutions that can cater to minorities in the demographics who are typically heavily under-represented in the training data.

{{</citation>}}


### (51/151) FLIP: Cross-domain Face Anti-spoofing with Language Guidance (Koushik Srivatsan et al., 2023)

{{<citation>}}

Koushik Srivatsan, Muzammal Naseer, Karthik Nandakumar. (2023)  
**FLIP: Cross-domain Face Anti-spoofing with Language Guidance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.16649v1)  

---


**ABSTRACT**  
Face anti-spoofing (FAS) or presentation attack detection is an essential component of face recognition systems deployed in security-critical applications. Existing FAS methods have poor generalizability to unseen spoof types, camera sensors, and environmental conditions. Recently, vision transformer (ViT) models have been shown to be effective for the FAS task due to their ability to capture long-range dependencies among image patches. However, adaptive modules or auxiliary loss functions are often required to adapt pre-trained ViT weights learned on large-scale datasets such as ImageNet. In this work, we first show that initializing ViTs with multimodal (e.g., CLIP) pre-trained weights improves generalizability for the FAS task, which is in line with the zero-shot transfer capabilities of vision-language pre-trained (VLP) models. We then propose a novel approach for robust cross-domain FAS by grounding visual representations with the help of natural language. Specifically, we show that aligning the image representation with an ensemble of class descriptions (based on natural language semantics) improves FAS generalizability in low-data regimes. Finally, we propose a multimodal contrastive learning strategy to boost feature generalization further and bridge the gap between source and target domains. Extensive experiments on three standard protocols demonstrate that our method significantly outperforms the state-of-the-art methods, achieving better zero-shot transfer performance than five-shot transfer of adaptive ViTs. Code: https://github.com/koushiksrivats/FLIP

{{</citation>}}


### (52/151) Improving Equivariance in State-of-the-Art Supervised Depth and Normal Predictors (Yuanyi Zhong et al., 2023)

{{<citation>}}

Yuanyi Zhong, Anand Bhattad, Yu-Xiong Wang, David Forsyth. (2023)  
**Improving Equivariance in State-of-the-Art Supervised Depth and Normal Predictors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.16646v1)  

---


**ABSTRACT**  
Dense depth and surface normal predictors should possess the equivariant property to cropping-and-resizing -- cropping the input image should result in cropping the same output image. However, we find that state-of-the-art depth and normal predictors, despite having strong performances, surprisingly do not respect equivariance. The problem exists even when crop-and-resize data augmentation is employed during training. To remedy this, we propose an equivariant regularization technique, consisting of an averaging procedure and a self-consistency loss, to explicitly promote cropping-and-resizing equivariance in depth and normal networks. Our approach can be applied to both CNN and Transformer architectures, does not incur extra cost during testing, and notably improves the supervised and semi-supervised learning performance of dense predictors on Taskonomy tasks. Finally, finetuning with our loss on unlabeled images improves not only equivariance but also accuracy of state-of-the-art depth and normal predictors when evaluated on NYU-v2. GitHub link: https://github.com/mikuhatsune/equivariance

{{</citation>}}


### (53/151) Deep Geometrized Cartoon Line Inbetweening (Li Siyao et al., 2023)

{{<citation>}}

Li Siyao, Tianpei Gu, Weiye Xiao, Henghui Ding, Ziwei Liu, Chen Change Loy. (2023)  
**Deep Geometrized Cartoon Line Inbetweening**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.16643v1)  

---


**ABSTRACT**  
We aim to address a significant but understudied problem in the anime industry, namely the inbetweening of cartoon line drawings. Inbetweening involves generating intermediate frames between two black-and-white line drawings and is a time-consuming and expensive process that can benefit from automation. However, existing frame interpolation methods that rely on matching and warping whole raster images are unsuitable for line inbetweening and often produce blurring artifacts that damage the intricate line structures. To preserve the precision and detail of the line drawings, we propose a new approach, AnimeInbet, which geometrizes raster line drawings into graphs of endpoints and reframes the inbetweening task as a graph fusion problem with vertex repositioning. Our method can effectively capture the sparsity and unique structure of line drawings while preserving the details during inbetweening. This is made possible via our novel modules, i.e., vertex geometric embedding, a vertex correspondence Transformer, an effective mechanism for vertex repositioning and a visibility predictor. To train our method, we introduce MixamoLine240, a new dataset of line drawings with ground truth vectorization and matching labels. Our experiments demonstrate that AnimeInbet synthesizes high-quality, clean, and complete intermediate line drawings, outperforming existing methods quantitatively and qualitatively, especially in cases with large motions. Data and code are available at https://github.com/lisiyao21/AnimeInbet.

{{</citation>}}


### (54/151) KV Inversion: KV Embeddings Learning for Text-Conditioned Real Image Action Editing (Jiancheng Huang et al., 2023)

{{<citation>}}

Jiancheng Huang, Yifan Liu, Jin Qin, Shifeng Chen. (2023)  
**KV Inversion: KV Embeddings Learning for Text-Conditioned Real Image Action Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.16608v1)  

---


**ABSTRACT**  
Text-conditioned image editing is a recently emerged and highly practical task, and its potential is immeasurable. However, most of the concurrent methods are unable to perform action editing, i.e. they can not produce results that conform to the action semantics of the editing prompt and preserve the content of the original image. To solve the problem of action editing, we propose KV Inversion, a method that can achieve satisfactory reconstruction performance and action editing, which can solve two major problems: 1) the edited result can match the corresponding action, and 2) the edited object can retain the texture and identity of the original real image. In addition, our method does not require training the Stable Diffusion model itself, nor does it require scanning a large-scale dataset to perform time-consuming training.

{{</citation>}}


### (55/151) Tensor Factorization for Leveraging Cross-Modal Knowledge in Data-Constrained Infrared Object Detection (Manish Sharma et al., 2023)

{{<citation>}}

Manish Sharma, Moitreya Chatterjee, Kuan-Chuan Peng, Suhas Lohit, Michael Jones. (2023)  
**Tensor Factorization for Leveraging Cross-Modal Knowledge in Data-Constrained Infrared Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.16592v1)  

---


**ABSTRACT**  
The primary bottleneck towards obtaining good recognition performance in IR images is the lack of sufficient labeled training data, owing to the cost of acquiring such data. Realizing that object detection methods for the RGB modality are quite robust (at least for some commonplace classes, like person, car, etc.), thanks to the giant training sets that exist, in this work we seek to leverage cues from the RGB modality to scale object detectors to the IR modality, while preserving model performance in the RGB modality. At the core of our method, is a novel tensor decomposition method called TensorFact which splits the convolution kernels of a layer of a Convolutional Neural Network (CNN) into low-rank factor matrices, with fewer parameters than the original CNN. We first pretrain these factor matrices on the RGB modality, for which plenty of training data are assumed to exist and then augment only a few trainable parameters for training on the IR modality to avoid over-fitting, while encouraging them to capture complementary cues from those trained only on the RGB modality. We validate our approach empirically by first assessing how well our TensorFact decomposed network performs at the task of detecting objects in RGB images vis-a-vis the original network and then look at how well it adapts to IR images of the FLIR ADAS v1 dataset. For the latter, we train models under scenarios that pose challenges stemming from data paucity. From the experiments, we observe that: (i) TensorFact shows performance gains on RGB images; (ii) further, this pre-trained model, when fine-tuned, outperforms a standard state-of-the-art object detector on the FLIR ADAS v1 dataset by about 4% in terms of mAP 50 score.

{{</citation>}}


### (56/151) Vision Transformers Need Registers (Timothée Darcet et al., 2023)

{{<citation>}}

Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski. (2023)  
**Vision Transformers Need Registers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16588v1)  

---


**ABSTRACT**  
Transformers have recently emerged as a powerful tool for learning visual representations. In this paper, we identify and characterize artifacts in feature maps of both supervised and self-supervised ViT networks. The artifacts correspond to high-norm tokens appearing during inference primarily in low-informative background areas of images, that are repurposed for internal computations. We propose a simple yet effective solution based on providing additional tokens to the input sequence of the Vision Transformer to fill that role. We show that this solution fixes that problem entirely for both supervised and self-supervised models, sets a new state of the art for self-supervised visual models on dense visual prediction tasks, enables object discovery methods with larger models, and most importantly leads to smoother feature maps and attention maps for downstream visual processing.

{{</citation>}}


### (57/151) MotionLM: Multi-Agent Motion Forecasting as Language Modeling (Ari Seff et al., 2023)

{{<citation>}}

Ari Seff, Brian Cera, Dian Chen, Mason Ng, Aurick Zhou, Nigamaa Nayakanti, Khaled S. Refaat, Rami Al-Rfou, Benjamin Sapp. (2023)  
**MotionLM: Multi-Agent Motion Forecasting as Language Modeling**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16534v1)  

---


**ABSTRACT**  
Reliable forecasting of the future behavior of road agents is a critical component to safe planning in autonomous vehicles. Here, we represent continuous trajectories as sequences of discrete motion tokens and cast multi-agent motion prediction as a language modeling task over this domain. Our model, MotionLM, provides several advantages: First, it does not require anchors or explicit latent variable optimization to learn multimodal distributions. Instead, we leverage a single standard language modeling objective, maximizing the average log probability over sequence tokens. Second, our approach bypasses post-hoc interaction heuristics where individual agent trajectory generation is conducted prior to interactive scoring. Instead, MotionLM produces joint distributions over interactive agent futures in a single autoregressive decoding process. In addition, the model's sequential factorization enables temporally causal conditional rollouts. The proposed approach establishes new state-of-the-art performance for multi-agent motion prediction on the Waymo Open Motion Dataset, ranking 1st on the interactive challenge leaderboard.

{{</citation>}}


### (58/151) Toloka Visual Question Answering Benchmark (Dmitry Ustalov et al., 2023)

{{<citation>}}

Dmitry Ustalov, Nikita Pavlichenko, Sergey Koshelev, Daniil Likhobaba, Alisa Smirnova. (2023)  
**Toloka Visual Question Answering Benchmark**  

---
Primary Category: cs.CV  
Categories: 68-11, C-4, cs-AI, cs-CL, cs-CV, cs-HC, cs.CV  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2309.16511v1)  

---


**ABSTRACT**  
In this paper, we present Toloka Visual Question Answering, a new crowdsourced dataset allowing comparing performance of machine learning systems against human level of expertise in the grounding visual question answering task. In this task, given an image and a textual question, one has to draw the bounding box around the object correctly responding to that question. Every image-question pair contains the response, with only one correct response per image. Our dataset contains 45,199 pairs of images and questions in English, provided with ground truth bounding boxes, split into train and two test subsets. Besides describing the dataset and releasing it under a CC BY license, we conducted a series of experiments on open source zero-shot baseline models and organized a multi-phase competition at WSDM Cup that attracted 48 participants worldwide. However, by the time of paper submission, no machine learning model outperformed the non-expert crowdsourcing baseline according to the intersection over union evaluation score.

{{</citation>}}


### (59/151) Rethinking Domain Generalization: Discriminability and Generalizability (Shaocong Long et al., 2023)

{{<citation>}}

Shaocong Long, Qianyu Zhou, Chenhao Ying, Lizhuang Ma, Yuan Luo. (2023)  
**Rethinking Domain Generalization: Discriminability and Generalizability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.16483v1)  

---


**ABSTRACT**  
Domain generalization (DG) endeavors to develop robust models that possess strong generalizability while preserving excellent discriminability. Nonetheless, pivotal DG techniques tend to improve the feature generalizability by learning domain-invariant representations, inadvertently overlooking the feature discriminability. On the one hand, the simultaneous attainment of generalizability and discriminability of features presents a complex challenge, often entailing inherent contradictions. This challenge becomes particularly pronounced when domain-invariant features manifest reduced discriminability owing to the inclusion of unstable factors, \emph{i.e.,} spurious correlations. On the other hand, prevailing domain-invariant methods can be categorized as category-level alignment, susceptible to discarding indispensable features possessing substantial generalizability and narrowing intra-class variations. To surmount these obstacles, we rethink DG from a new perspective that concurrently imbues features with formidable discriminability and robust generalizability, and present a novel framework, namely, Discriminative Microscopic Distribution Alignment (DMDA). DMDA incorporates two core components: Selective Channel Pruning~(SCP) and Micro-level Distribution Alignment (MDA). Concretely, SCP attempts to curtail redundancy within neural networks, prioritizing stable attributes conducive to accurate classification. This approach alleviates the adverse effect of spurious domain invariance and amplifies the feature discriminability. Besides, MDA accentuates micro-level alignment within each class, going beyond mere category-level alignment. This strategy accommodates sufficient generalizable features and facilitates within-class variations. Extensive experiments on four benchmark datasets corroborate the efficacy of our method.

{{</citation>}}


### (60/151) Radar Instance Transformer: Reliable Moving Instance Segmentation in Sparse Radar Point Clouds (Matthias Zeller et al., 2023)

{{<citation>}}

Matthias Zeller, Vardeep S. Sandhu, Benedikt Mersch, Jens Behley, Michael Heidingsfeld, Cyrill Stachniss. (2023)  
**Radar Instance Transformer: Reliable Moving Instance Segmentation in Sparse Radar Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.16435v1)  

---


**ABSTRACT**  
The perception of moving objects is crucial for autonomous robots performing collision avoidance in dynamic environments. LiDARs and cameras tremendously enhance scene interpretation but do not provide direct motion information and face limitations under adverse weather. Radar sensors overcome these limitations and provide Doppler velocities, delivering direct information on dynamic objects. In this paper, we address the problem of moving instance segmentation in radar point clouds to enhance scene interpretation for safety-critical tasks. Our Radar Instance Transformer enriches the current radar scan with temporal information without passing aggregated scans through a neural network. We propose a full-resolution backbone to prevent information loss in sparse point cloud processing. Our instance transformer head incorporates essential information to enhance segmentation but also enables reliable, class-agnostic instance assignments. In sum, our approach shows superior performance on the new moving instance segmentation benchmarks, including diverse environments, and provides model-agnostic modules to enhance scene interpretation. The benchmark is based on the RadarScenes dataset and will be made available upon acceptance.

{{</citation>}}


### (61/151) AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models (Jan Hendrik Metzen et al., 2023)

{{<citation>}}

Jan Hendrik Metzen, Piyapat Saranrittichai, Chaithanya Kumar Mummadi. (2023)  
**AutoCLIP: Auto-tuning Zero-Shot Classifiers for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.16414v2)  

---


**ABSTRACT**  
Classifiers built upon vision-language models such as CLIP have shown remarkable zero-shot performance across a broad range of image classification tasks. Prior work has studied different ways of automatically creating descriptor sets for every class based on prompt templates, ranging from manually engineered templates over templates obtained from a large language model to templates built from random words and characters. Up until now, deriving zero-shot classifiers from the respective encoded class descriptors has remained nearly unchanged, i.e., classify to the class that maximizes cosine similarity between its averaged encoded class descriptors and the image encoding. However, weighing all class descriptors equally can be suboptimal when certain descriptors match visual clues on a given image better than others. In this work, we propose AutoCLIP, a method for auto-tuning zero-shot classifiers. AutoCLIP tunes per-image weights to each prompt template at inference time, based on statistics of class descriptor-image similarities. AutoCLIP is fully unsupervised, has very low computational overhead, and can be easily implemented in few lines of code. We show that AutoCLIP outperforms baselines across a broad range of vision-language models, datasets, and prompt templates consistently and by up to 3 percent point accuracy.

{{</citation>}}


### (62/151) HIC-YOLOv5: Improved YOLOv5 For Small Object Detection (Shiyi Tang et al., 2023)

{{<citation>}}

Shiyi Tang, Yini Fang, Shu Zhang. (2023)  
**HIC-YOLOv5: Improved YOLOv5 For Small Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.16393v1)  

---


**ABSTRACT**  
Small object detection has been a challenging problem in the field of object detection. There has been some works that proposes improvements for this task, such as adding several attention blocks or changing the whole structure of feature fusion networks. However, the computation cost of these models is large, which makes deploying a real-time object detection system unfeasible, while leaving room for improvement. To this end, an improved YOLOv5 model: HIC-YOLOv5 is proposed to address the aforementioned problems. Firstly, an additional prediction head specific to small objects is added to provide a higher-resolution feature map for better prediction. Secondly, an involution block is adopted between the backbone and neck to increase channel information of the feature map. Moreover, an attention mechanism named CBAM is applied at the end of the backbone, thus not only decreasing the computation cost compared with previous works but also emphasizing the important information in both channel and spatial domain. Our result shows that HIC-YOLOv5 has improved mAP@[.5:.95] by 6.42% and mAP@0.5 by 9.38% on VisDrone-2019-DET dataset.

{{</citation>}}


### (63/151) Dark Side Augmentation: Generating Diverse Night Examples for Metric Learning (Albert Mohwald et al., 2023)

{{<citation>}}

Albert Mohwald, Tomas Jenicek, Ondřej Chum. (2023)  
**Dark Side Augmentation: Generating Diverse Night Examples for Metric Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.16351v1)  

---


**ABSTRACT**  
Image retrieval methods based on CNN descriptors rely on metric learning from a large number of diverse examples of positive and negative image pairs. Domains, such as night-time images, with limited availability and variability of training data suffer from poor retrieval performance even with methods performing well on standard benchmarks. We propose to train a GAN-based synthetic-image generator, translating available day-time image examples into night images. Such a generator is used in metric learning as a form of augmentation, supplying training data to the scarce domain. Various types of generators are evaluated and analyzed. We contribute with a novel light-weight GAN architecture that enforces the consistency between the original and translated image through edge consistency. The proposed architecture also allows a simultaneous training of an edge detector that operates on both night and day images. To further increase the variability in the training examples and to maximize the generalization of the trained model, we propose a novel method of diverse anchor mining.   The proposed method improves over the state-of-the-art results on a standard Tokyo 24/7 day-night retrieval benchmark while preserving the performance on Oxford and Paris datasets. This is achieved without the need of training image pairs of matching day and night images. The source code is available at https://github.com/mohwald/gandtr .

{{</citation>}}


### (64/151) Logarithm-transform aided Gaussian Sampling for Few-Shot Learning (Vaibhav Ganatra, 2023)

{{<citation>}}

Vaibhav Ganatra. (2023)  
**Logarithm-transform aided Gaussian Sampling for Few-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.16337v1)  

---


**ABSTRACT**  
Few-shot image classification has recently witnessed the rise of representation learning being utilised for models to adapt to new classes using only a few training examples. Therefore, the properties of the representations, such as their underlying probability distributions, assume vital importance. Representations sampled from Gaussian distributions have been used in recent works, [19] to train classifiers for few-shot classification. These methods rely on transforming the distributions of experimental data to approximate Gaussian distributions for their functioning. In this paper, I propose a novel Gaussian transform, that outperforms existing methods on transforming experimental data into Gaussian-like distributions. I then utilise this novel transformation for few-shot image classification and show significant gains in performance, while sampling lesser data.

{{</citation>}}


### (65/151) Weakly-Supervised Video Anomaly Detection with Snippet Anomalous Attention (Yidan Fan et al., 2023)

{{<citation>}}

Yidan Fan, Yongxin Yu, Wenhuan Lu, Yahong Han. (2023)  
**Weakly-Supervised Video Anomaly Detection with Snippet Anomalous Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2309.16309v1)  

---


**ABSTRACT**  
With a focus on abnormal events contained within untrimmed videos, there is increasing interest among researchers in video anomaly detection. Among different video anomaly detection scenarios, weakly-supervised video anomaly detection poses a significant challenge as it lacks frame-wise labels during the training stage, only relying on video-level labels as coarse supervision. Previous methods have made attempts to either learn discriminative features in an end-to-end manner or employ a twostage self-training strategy to generate snippet-level pseudo labels. However, both approaches have certain limitations. The former tends to overlook informative features at the snippet level, while the latter can be susceptible to noises. In this paper, we propose an Anomalous Attention mechanism for weakly-supervised anomaly detection to tackle the aforementioned problems. Our approach takes into account snippet-level encoded features without the supervision of pseudo labels. Specifically, our approach first generates snippet-level anomalous attention and then feeds it together with original anomaly scores into a Multi-branch Supervision Module. The module learns different areas of the video, including areas that are challenging to detect, and also assists the attention optimization. Experiments on benchmark datasets XDViolence and UCF-Crime verify the effectiveness of our method. Besides, thanks to the proposed snippet-level attention, we obtain a more precise anomaly localization.

{{</citation>}}


### (66/151) Multi-scale Recurrent LSTM and Transformer Network for Depth Completion (Xiaogang Jia et al., 2023)

{{<citation>}}

Xiaogang Jia, Yusong Tan, Songlei Jian, Yonggang Che. (2023)  
**Multi-scale Recurrent LSTM and Transformer Network for Depth Completion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16301v1)  

---


**ABSTRACT**  
Lidar depth completion is a new and hot topic of depth estimation. In this task, it is the key and difficult point to fuse the features of color space and depth space. In this paper, we migrate the classic LSTM and Transformer modules from NLP to depth completion and redesign them appropriately. Specifically, we use Forget gate, Update gate, Output gate, and Skip gate to achieve the efficient fusion of color and depth features and perform loop optimization at multiple scales. Finally, we further fuse the deep features through the Transformer multi-head attention mechanism. Experimental results show that without repetitive network structure and post-processing steps, our method can achieve state-of-the-art performance by adding our modules to a simple encoder-decoder network structure. Our method ranks first on the current mainstream autonomous driving KITTI benchmark dataset. It can also be regarded as a backbone network for other methods, which likewise achieves state-of-the-art performance.

{{</citation>}}


### (67/151) FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding (Pengxiang Wu et al., 2023)

{{<citation>}}

Pengxiang Wu, Siman Wang, Kevin Dela Rosa, Derek Hao Hu. (2023)  
**FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.16249v1)  

---


**ABSTRACT**  
Image retrieval is a fundamental task in computer vision. Despite recent advances in this field, many techniques have been evaluated on a limited number of domains, with a small number of instance categories. Notably, most existing works only consider domains like 3D landmarks, making it difficult to generalize the conclusions made by these works to other domains, e.g., logo and other 2D flat objects. To bridge this gap, we introduce a new dataset for benchmarking visual search methods on flat images with diverse patterns. Our flat object retrieval benchmark (FORB) supplements the commonly adopted 3D object domain, and more importantly, it serves as a testbed for assessing the image embedding quality on out-of-distribution domains. In this benchmark we investigate the retrieval accuracy of representative methods in terms of candidate ranks, as well as matching score margin, a viewpoint which is largely ignored by many works. Our experiments not only highlight the challenges and rich heterogeneity of FORB, but also reveal the hidden properties of different retrieval strategies. The proposed benchmark is a growing project and we expect to expand in both quantity and variety of objects. The dataset and supporting codes are available at https://github.com/pxiangwu/FORB/.

{{</citation>}}


### (68/151) Object Motion Guided Human Motion Synthesis (Jiaman Li et al., 2023)

{{<citation>}}

Jiaman Li, Jiajun Wu, C. Karen Liu. (2023)  
**Object Motion Guided Human Motion Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16237v1)  

---


**ABSTRACT**  
Modeling human behaviors in contextual environments has a wide range of applications in character animation, embodied AI, VR/AR, and robotics. In real-world scenarios, humans frequently interact with the environment and manipulate various objects to complete daily tasks. In this work, we study the problem of full-body human motion synthesis for the manipulation of large-sized objects. We propose Object MOtion guided human MOtion synthesis (OMOMO), a conditional diffusion framework that can generate full-body manipulation behaviors from only the object motion. Since naively applying diffusion models fails to precisely enforce contact constraints between the hands and the object, OMOMO learns two separate denoising processes to first predict hand positions from object motion and subsequently synthesize full-body poses based on the predicted hand positions. By employing the hand positions as an intermediate representation between the two denoising processes, we can explicitly enforce contact constraints, resulting in more physically plausible manipulation motions. With the learned model, we develop a novel system that captures full-body human manipulation motions by simply attaching a smartphone to the object being manipulated. Through extensive experiments, we demonstrate the effectiveness of our proposed pipeline and its ability to generalize to unseen objects. Additionally, as high-quality human-object interaction datasets are scarce, we collect a large-scale dataset consisting of 3D object geometry, object motion, and human motion. Our dataset contains human-object interaction motion for 15 objects, with a total duration of approximately 10 hours.

{{</citation>}}


### (69/151) GAFlow: Incorporating Gaussian Attention into Optical Flow (Ao Luo et al., 2023)

{{<citation>}}

Ao Luo, Fan Yang, Xin Li, Lang Nie, Chunyu Lin, Haoqiang Fan, Shuaicheng Liu. (2023)  
**GAFlow: Incorporating Gaussian Attention into Optical Flow**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16217v1)  

---


**ABSTRACT**  
Optical flow, or the estimation of motion fields from image sequences, is one of the fundamental problems in computer vision. Unlike most pixel-wise tasks that aim at achieving consistent representations of the same category, optical flow raises extra demands for obtaining local discrimination and smoothness, which yet is not fully explored by existing approaches. In this paper, we push Gaussian Attention (GA) into the optical flow models to accentuate local properties during representation learning and enforce the motion affinity during matching. Specifically, we introduce a novel Gaussian-Constrained Layer (GCL) which can be easily plugged into existing Transformer blocks to highlight the local neighborhood that contains fine-grained structural information. Moreover, for reliable motion analysis, we provide a new Gaussian-Guided Attention Module (GGAM) which not only inherits properties from Gaussian distribution to instinctively revolve around the neighbor fields of each point but also is empowered to put the emphasis on contextually related regions during matching. Our fully-equipped model, namely Gaussian Attention Flow network (GAFlow), naturally incorporates a series of novel Gaussian-based modules into the conventional optical flow framework for reliable motion analysis. Extensive experiments on standard optical flow datasets consistently demonstrate the exceptional performance of the proposed approach in terms of both generalization ability evaluation and online benchmark testing. Code is available at https://github.com/LA30/GAFlow.

{{</citation>}}


### (70/151) VDC: Versatile Data Cleanser for Detecting Dirty Samples via Visual-Linguistic Inconsistency (Zihao Zhu et al., 2023)

{{<citation>}}

Zihao Zhu, Mingda Zhang, Shaokui Wei, Bingzhe Wu, Baoyuan Wu. (2023)  
**VDC: Versatile Data Cleanser for Detecting Dirty Samples via Visual-Linguistic Inconsistency**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16211v1)  

---


**ABSTRACT**  
The role of data in building AI systems has recently been emphasized by the emerging concept of data-centric AI. Unfortunately, in the real-world, datasets may contain dirty samples, such as poisoned samples from backdoor attack, noisy labels in crowdsourcing, and even hybrids of them. The presence of such dirty samples makes the DNNs vunerable and unreliable.Hence, it is critical to detect dirty samples to improve the quality and realiability of dataset. Existing detectors only focus on detecting poisoned samples or noisy labels, that are often prone to weak generalization when dealing with dirty samples from other domains.In this paper, we find a commonality of various dirty samples is visual-linguistic inconsistency between images and associated labels. To capture the semantic inconsistency between modalities, we propose versatile data cleanser (VDC) leveraging the surpassing capabilities of multimodal large language models (MLLM) in cross-modal alignment and reasoning.It consists of three consecutive modules: the visual question generation module to generate insightful questions about the image; the visual question answering module to acquire the semantics of the visual content by answering the questions with MLLM; followed by the visual answer evaluation module to evaluate the inconsistency.Extensive experiments demonstrate its superior performance and generalization to various categories and types of dirty samples.

{{</citation>}}


### (71/151) Parameter-Saving Adversarial Training: Reinforcing Multi-Perturbation Robustness via Hypernetworks (Huihui Gong et al., 2023)

{{<citation>}}

Huihui Gong, Minjing Dong, Siqi Ma, Seyit Camtepe, Surya Nepal, Chang Xu. (2023)  
**Parameter-Saving Adversarial Training: Reinforcing Multi-Perturbation Robustness via Hypernetworks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2309.16207v1)  

---


**ABSTRACT**  
Adversarial training serves as one of the most popular and effective methods to defend against adversarial perturbations. However, most defense mechanisms only consider a single type of perturbation while various attack methods might be adopted to perform stronger adversarial attacks against the deployed model in real-world scenarios, e.g., $\ell_2$ or $\ell_\infty$. Defending against various attacks can be a challenging problem since multi-perturbation adversarial training and its variants only achieve suboptimal robustness trade-offs, due to the theoretical limit to multi-perturbation robustness for a single model. Besides, it is impractical to deploy large models in some storage-efficient scenarios. To settle down these drawbacks, in this paper we propose a novel multi-perturbation adversarial training framework, parameter-saving adversarial training (PSAT), to reinforce multi-perturbation robustness with an advantageous side effect of saving parameters, which leverages hypernetworks to train specialized models against a single perturbation and aggregate these specialized models to defend against multiple perturbations. Eventually, we extensively evaluate and compare our proposed method with state-of-the-art single/multi-perturbation robust methods against various latest attack methods on different datasets, showing the robustness superiority and parameter efficiency of our proposed method, e.g., for the CIFAR-10 dataset with ResNet-50 as the backbone, PSAT saves approximately 80\% of parameters with achieving the state-of-the-art robustness trade-off accuracy.

{{</citation>}}


### (72/151) BEVHeight++: Toward Robust Visual Centric 3D Object Detection (Lei Yang et al., 2023)

{{<citation>}}

Lei Yang, Tao Tang, Jun Li, Peng Chen, Kun Yuan, Li Wang, Yi Huang, Xinyu Zhang, Kaicheng Yu. (2023)  
**BEVHeight++: Toward Robust Visual Centric 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.16179v1)  

---


**ABSTRACT**  
While most recent autonomous driving system focuses on developing perception methods on ego-vehicle sensors, people tend to overlook an alternative approach to leverage intelligent roadside cameras to extend the perception ability beyond the visual range. We discover that the state-of-the-art vision-centric bird's eye view detection methods have inferior performances on roadside cameras. This is because these methods mainly focus on recovering the depth regarding the camera center, where the depth difference between the car and the ground quickly shrinks while the distance increases. In this paper, we propose a simple yet effective approach, dubbed BEVHeight++, to address this issue. In essence, we regress the height to the ground to achieve a distance-agnostic formulation to ease the optimization process of camera-only perception methods. By incorporating both height and depth encoding techniques, we achieve a more accurate and robust projection from 2D to BEV spaces. On popular 3D detection benchmarks of roadside cameras, our method surpasses all previous vision-centric methods by a significant margin. In terms of the ego-vehicle scenario, our BEVHeight++ possesses superior over depth-only methods. Specifically, it yields a notable improvement of +1.9% NDS and +1.1% mAP over BEVDepth when evaluated on the nuScenes validation set. Moreover, on the nuScenes test set, our method achieves substantial advancements, with an increase of +2.8% NDS and +1.7% mAP, respectively.

{{</citation>}}


### (73/151) ELIP: Efficient Language-Image Pre-training with Fewer Vision Tokens (Yangyang Guo et al., 2023)

{{<citation>}}

Yangyang Guo, Haoyu Zhang, Liqiang Nie, Yongkang Wong, Mohan Kankanhalli. (2023)  
**ELIP: Efficient Language-Image Pre-training with Fewer Vision Tokens**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.16738v1)  

---


**ABSTRACT**  
Learning a versatile language-image model is computationally prohibitive under a limited computing budget. This paper delves into the efficient language-image pre-training, an area that has received relatively little attention despite its importance in reducing computational cost and footprint. To that end, we propose a vision token pruning and merging method, ie ELIP, to remove less influential tokens based on the supervision of language outputs. Our method is designed with several strengths, such as being computation-efficient, memory-efficient, and trainable-parameter-free, and is distinguished from previous vision-only token pruning approaches by its alignment with task objectives. We implement this method in a progressively pruning manner using several sequential blocks. To evaluate its generalization performance, we apply ELIP to three commonly used language-image pre-training models and utilize public image-caption pairs with 4M images for pre-training. Our experiments demonstrate that with the removal of ~30$\%$ vision tokens across 12 ViT layers, ELIP maintains significantly comparable performance with baselines ($\sim$0.32 accuracy drop on average) over various downstream tasks including cross-modal retrieval, VQA, image captioning, etc. In addition, the spared GPU resources by our ELIP allow us to scale up with larger batch sizes, thereby accelerating model pre-training and even sometimes enhancing downstream model performance. Our code will be released at https://github.com/guoyang9/ELIP.

{{</citation>}}


### (74/151) Two-Step Active Learning for Instance Segmentation with Uncertainty and Diversity Sampling (Ke Yu et al., 2023)

{{<citation>}}

Ke Yu, Stephen Albro, Giulia DeSalvo, Suraj Kothawade, Abdullah Rashwan, Sasan Tavakkol, Kayhan Batmanghelich, Xiaoqi Yin. (2023)  
**Two-Step Active Learning for Instance Segmentation with Uncertainty and Diversity Sampling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.16139v1)  

---


**ABSTRACT**  
Training high-quality instance segmentation models requires an abundance of labeled images with instance masks and classifications, which is often expensive to procure. Active learning addresses this challenge by striving for optimum performance with minimal labeling cost by selecting the most informative and representative images for labeling. Despite its potential, active learning has been less explored in instance segmentation compared to other tasks like image classification, which require less labeling. In this study, we propose a post-hoc active learning algorithm that integrates uncertainty-based sampling with diversity-based sampling. Our proposed algorithm is not only simple and easy to implement, but it also delivers superior performance on various datasets. Its practical application is demonstrated on a real-world overhead imagery dataset, where it increases the labeling efficiency fivefold.

{{</citation>}}


### (75/151) Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval (Yuanmin Tang et al., 2023)

{{<citation>}}

Yuanmin Tang, Jing Yu, Keke Gai, Zhuang Jiamin, Gang Xiong, Yue Hu, Qi Wu. (2023)  
**Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.16137v1)  

---


**ABSTRACT**  
Different from Composed Image Retrieval task that requires expensive labels for training task-specific models, Zero-Shot Composed Image Retrieval (ZS-CIR) involves diverse tasks with a broad range of visual content manipulation intent that could be related to domain, scene, object, and attribute. The key challenge for ZS-CIR tasks is to learn a more accurate image representation that has adaptive attention to the reference image for various manipulation descriptions. In this paper, we propose a novel context-dependent mapping network, named Context-I2W, for adaptively converting description-relevant Image information into a pseudo-word token composed of the description for accurate ZS-CIR. Specifically, an Intent View Selector first dynamically learns a rotation rule to map the identical image to a task-specific manipulation view. Then a Visual Target Extractor further captures local information covering the main targets in ZS-CIR tasks under the guidance of multiple learnable queries. The two complementary modules work together to map an image to a context-dependent pseudo-word token without extra supervision. Our model shows strong generalization ability on four ZS-CIR tasks, including domain conversion, object composition, object manipulation, and attribute manipulation. It obtains consistent and significant performance boosts ranging from 1.88% to 3.60% over the best methods and achieves new state-of-the-art results on ZS-CIR. Our code is available at https://github.com/Pter61/context_i2w.

{{</citation>}}


### (76/151) A dual-branch model with inter- and intra-branch contrastive loss for long-tailed recognition (Qiong Chen et al., 2023)

{{<citation>}}

Qiong Chen, Tianlin Huang, Geren Zhu, Enlu Lin. (2023)  
**A dual-branch model with inter- and intra-branch contrastive loss for long-tailed recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, ImageNet  
[Paper Link](http://arxiv.org/abs/2309.16135v1)  

---


**ABSTRACT**  
Real-world data often exhibits a long-tailed distribution, in which head classes occupy most of the data, while tail classes only have very few samples. Models trained on long-tailed datasets have poor adaptability to tail classes and the decision boundaries are ambiguous. Therefore, in this paper, we propose a simple yet effective model, named Dual-Branch Long-Tailed Recognition (DB-LTR), which includes an imbalanced learning branch and a Contrastive Learning Branch (CoLB). The imbalanced learning branch, which consists of a shared backbone and a linear classifier, leverages common imbalanced learning approaches to tackle the data imbalance issue. In CoLB, we learn a prototype for each tail class, and calculate an inter-branch contrastive loss, an intra-branch contrastive loss and a metric loss. CoLB can improve the capability of the model in adapting to tail classes and assist the imbalanced learning branch to learn a well-represented feature space and discriminative decision boundary. Extensive experiments on three long-tailed benchmark datasets, i.e., CIFAR100-LT, ImageNet-LT and Places-LT, show that our DB-LTR is competitive and superior to the comparative methods.

{{</citation>}}


### (77/151) MASK4D: Mask Transformer for 4D Panoptic Segmentation (Kadir Yilmaz et al., 2023)

{{<citation>}}

Kadir Yilmaz, Jonas Schult, Alexey Nekrasov, Bastian Leibe. (2023)  
**MASK4D: Mask Transformer for 4D Panoptic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.16133v1)  

---


**ABSTRACT**  
Accurately perceiving and tracking instances over time is essential for the decision-making processes of autonomous agents interacting safely in dynamic environments. With this intention, we propose Mask4D for the challenging task of 4D panoptic segmentation of LiDAR point clouds. Mask4D is the first transformer-based approach unifying semantic instance segmentation and tracking of sparse and irregular sequences of 3D point clouds into a single joint model. Our model directly predicts semantic instances and their temporal associations without relying on any hand-crafted non-learned association strategies such as probabilistic clustering or voting-based center prediction. Instead, Mask4D introduces spatio-temporal instance queries which encode the semantic and geometric properties of each semantic tracklet in the sequence. In an in-depth study, we find that it is critical to promote spatially compact instance predictions as spatio-temporal instance queries tend to merge multiple semantically similar instances, even if they are spatially distant. To this end, we regress 6-DOF bounding box parameters from spatio-temporal instance queries, which is used as an auxiliary task to foster spatially compact predictions. Mask4D achieves a new state-of-the-art on the SemanticKITTI test set with a score of 68.4 LSTQ, improving upon published top-performing methods by at least +4.5%.

{{</citation>}}


### (78/151) Open Compound Domain Adaptation with Object Style Compensation for Semantic Segmentation (Tingliang Feng et al., 2023)

{{<citation>}}

Tingliang Feng, Hao Shi, Xueyang Liu, Wei Feng, Liang Wan, Yanlin Zhou, Di Lin. (2023)  
**Open Compound Domain Adaptation with Object Style Compensation for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.16127v1)  

---


**ABSTRACT**  
Many methods of semantic image segmentation have borrowed the success of open compound domain adaptation. They minimize the style gap between the images of source and target domains, more easily predicting the accurate pseudo annotations for target domain's images that train segmentation network. The existing methods globally adapt the scene style of the images, whereas the object styles of different categories or instances are adapted improperly. This paper proposes the Object Style Compensation, where we construct the Object-Level Discrepancy Memory with multiple sets of discrepancy features. The discrepancy features in a set capture the style changes of the same category's object instances adapted from target to source domains. We learn the discrepancy features from the images of source and target domains, storing the discrepancy features in memory. With this memory, we select appropriate discrepancy features for compensating the style information of the object instances of various categories, adapting the object styles to a unified style of source domain. Our method enables a more accurate computation of the pseudo annotations for target domain's images, thus yielding state-of-the-art results on different datasets.

{{</citation>}}


### (79/151) Channel Vision Transformers: An Image Is Worth C x 16 x 16 Words (Yujia Bao et al., 2023)

{{<citation>}}

Yujia Bao, Srinivasan Sivanandan, Theofanis Karaletsos. (2023)  
**Channel Vision Transformers: An Image Is Worth C x 16 x 16 Words**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16108v1)  

---


**ABSTRACT**  
Vision Transformer (ViT) has emerged as a powerful architecture in the realm of modern computer vision. However, its application in certain imaging fields, such as microscopy and satellite imaging, presents unique challenges. In these domains, images often contain multiple channels, each carrying semantically distinct and independent information. Furthermore, the model must demonstrate robustness to sparsity in input channels, as they may not be densely available during training or testing. In this paper, we propose a modification to the ViT architecture that enhances reasoning across the input channels and introduce Hierarchical Channel Sampling (HCS) as an additional regularization technique to ensure robustness when only partial channels are presented during test time. Our proposed model, ChannelViT, constructs patch tokens independently from each input channel and utilizes a learnable channel embedding that is added to the patch tokens, similar to positional embeddings. We evaluate the performance of ChannelViT on ImageNet, JUMP-CP (microscopy cell imaging), and So2Sat (satellite imaging). Our results show that ChannelViT outperforms ViT on classification tasks and generalizes well, even when a subset of input channels is used during testing. Across our experiments, HCS proves to be a powerful regularizer, independent of the architecture employed, suggesting itself as a straightforward technique for robust ViT training. Lastly, we find that ChannelViT generalizes effectively even when there is limited access to all channels during training, highlighting its potential for multi-channel imaging under real-world conditions with sparse sensors.

{{</citation>}}


## cs.NI (2)



### (80/151) Vidaptive: Efficient and Responsive Rate Control for Real-Time Video on Variable Networks (Pantea Karimi et al., 2023)

{{<citation>}}

Pantea Karimi, Sadjad Fouladi, Vibhaalakshmi Sivaraman, Mohammad Alizadeh. (2023)  
**Vidaptive: Efficient and Responsive Rate Control for Real-Time Video on Variable Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.16869v1)  

---


**ABSTRACT**  
Real-time video streaming relies on rate control mechanisms to adapt video bitrate to network capacity while maintaining high utilization and low delay. However, the current video rate controllers, such as Google Congestion Control (GCC) in WebRTC, are very slow to respond to network changes, leading to link under-utilization and latency spikes. While recent delay-based congestion control algorithms promise high efficiency and rapid adaptation to variable conditions, low-latency video applications have been unable to adopt these schemes due to the intertwined relationship between video encoders and rate control in current systems.   This paper introduces Vidaptive, a new rate control mechanism designed for low-latency video applications. Vidaptive decouples packet transmission decisions from encoder output, injecting dummy padding traffic as needed to treat video streams akin to backlogged flows controlled by a delay-based congestion controller. Vidaptive then adapts the frame rate, resolution, and target bitrate of the encoder to align the video bitrate with the congestion controller's sending rate. Our evaluations atop WebRTC show that, across a set of cellular traces, Vidaptive achieves ~2x higher video bitrate and 1.6 dB higher PSNR, and it reduces 95th-percentile frame latency by 2.7s with a slight increase in median frame latency.

{{</citation>}}


### (81/151) Libertas: Privacy-Preserving Computation for Decentralised Personal Data Stores (Rui Zhao et al., 2023)

{{<citation>}}

Rui Zhao, Naman Goel, Nitin Agrawal, Jun Zhao, Jake Stein, Ruben Verborgh, Reuben Binns, Tim Berners-Lee, Nigel Shadbolt. (2023)  
**Libertas: Privacy-Preserving Computation for Decentralised Personal Data Stores**  

---
Primary Category: cs.NI  
Categories: cs-CR, cs-DC, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16365v1)  

---


**ABSTRACT**  
Data-driven decision-making and AI applications present exciting new opportunities delivering widespread benefits. The rapid adoption of such applications triggers legitimate concerns about loss of privacy and misuse of personal data. This leads to a growing and pervasive tension between harvesting ubiquitous data on the Web and the need to protect individuals. Decentralised personal data stores (PDS) such as Solid are frameworks designed to give individuals ultimate control over their personal data. But current PDS approaches have limited support for ensuring privacy when computations combine data spread across users. Secure Multi-Party Computation (MPC) is a well-known subfield of cryptography, enabling multiple autonomous parties to collaboratively compute a function while ensuring the secrecy of inputs (input privacy). These two technologies complement each other, but existing practices fall short in addressing the requirements and challenges of introducing MPC in a PDS environment. For the first time, we propose a modular design for integrating MPC with Solid while respecting the requirements of decentralisation in this context. Our architecture, Libertas, requires no protocol level changes in the underlying design of Solid, and can be adapted to other PDS. We further show how this can be combined with existing differential privacy techniques to also ensure output privacy. We use empirical benchmarks to inform and evaluate our implementation and design choices. We show the technical feasibility and scalability pattern of the proposed system in two novel scenarios -- 1) empowering gig workers with aggregate computations on their earnings data; and 2) generating high-quality differentially-private synthetic data without requiring a trusted centre. With this, we demonstrate the linear scalability of Libertas, and gained insights about compute optimisations under such an architecture.

{{</citation>}}


## cs.CL (35)



### (82/151) DeBERTinha: A Multistep Approach to Adapt DebertaV3 XSmall for Brazilian Portuguese Natural Language Processing Task (Israel Campiotti et al., 2023)

{{<citation>}}

Israel Campiotti, Matheus Rodrigues, Yuri Albuquerque, Rafael Azevedo, Alyson Andrade. (2023)  
**DeBERTinha: A Multistep Approach to Adapt DebertaV3 XSmall for Brazilian Portuguese Natural Language Processing Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.16844v1)  

---


**ABSTRACT**  
This paper presents an approach for adapting the DebertaV3 XSmall model pre-trained in English for Brazilian Portuguese natural language processing (NLP) tasks. A key aspect of the methodology involves a multistep training process to ensure the model is effectively tuned for the Portuguese language. Initial datasets from Carolina and BrWac are preprocessed to address issues like emojis, HTML tags, and encodings. A Portuguese-specific vocabulary of 50,000 tokens is created using SentencePiece. Rather than training from scratch, the weights of the pre-trained English model are used to initialize most of the network, with random embeddings, recognizing the expensive cost of training from scratch. The model is fine-tuned using the replaced token detection task in the same format of DebertaV3 training. The adapted model, called DeBERTinha, demonstrates effectiveness on downstream tasks like named entity recognition, sentiment analysis, and determining sentence relatedness, outperforming BERTimbau-Large in two tasks despite having only 40M parameters.

{{</citation>}}


### (83/151) Curriculum-Driven Edubot: A Framework for Developing Language Learning Chatbots Through Synthesizing Conversational Data (Yu Li et al., 2023)

{{<citation>}}

Yu Li, Shang Qu, Jili Shen, Shangchao Min, Zhou Yu. (2023)  
**Curriculum-Driven Edubot: A Framework for Developing Language Learning Chatbots Through Synthesizing Conversational Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.16804v1)  

---


**ABSTRACT**  
Chatbots have become popular in educational settings, revolutionizing how students interact with material and how teachers teach. We present Curriculum-Driven EduBot, a framework for developing a chatbot that combines the interactive features of chatbots with the systematic material of English textbooks to assist students in enhancing their conversational skills. We begin by extracting pertinent topics from textbooks and then using large language models to generate dialogues related to these topics. We then fine-tune an open-source LLM using our generated conversational data to create our curriculum-driven chatbot. User studies demonstrate that our chatbot outperforms ChatGPT in leading curriculum-based dialogues and adapting its dialogue to match the user's English proficiency level. By combining traditional textbook methodologies with conversational AI, our approach offers learners an interactive tool that aligns with their curriculum and provides user-tailored conversation practice. This facilitates meaningful student-bot dialogues and enriches the overall learning experience within the curriculum's pedagogical framework.

{{</citation>}}


### (84/151) Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution (Chrisantha Fernando et al., 2023)

{{<citation>}}

Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, Tim Rocktäschel. (2023)  
**Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16797v1)  

---


**ABSTRACT**  
Popular prompt strategies like Chain-of-Thought Prompting can dramatically improve the reasoning abilities of Large Language Models (LLMs) in various domains. However, such hand-crafted prompt-strategies are often sub-optimal. In this paper, we present Promptbreeder, a general-purpose self-referential self-improvement mechanism that evolves and adapts prompts for a given domain. Driven by an LLM, Promptbreeder mutates a population of task-prompts, and subsequently evaluates them for fitness on a training set. Crucially, the mutation of these task-prompts is governed by mutation-prompts that the LLM generates and improves throughout evolution in a self-referential way. That is, Promptbreeder is not just improving task-prompts, but it is also improving the mutationprompts that improve these task-prompts. Promptbreeder outperforms state-of-the-art prompt strategies such as Chain-of-Thought and Plan-and-Solve Prompting on commonly used arithmetic and commonsense reasoning benchmarks. Furthermore, Promptbreeder is able to evolve intricate task-prompts for the challenging problem of hate speech classification.

{{</citation>}}


### (85/151) Hallucination Reduction in Long Input Text Summarization (Tohida Rehman et al., 2023)

{{<citation>}}

Tohida Rehman, Ronit Mandal, Abhishek Agarwal, Debarshi Kumar Sanyal. (2023)  
**Hallucination Reduction in Long Input Text Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2309.16781v1)  

---


**ABSTRACT**  
Hallucination in text summarization refers to the phenomenon where the model generates information that is not supported by the input source document. Hallucination poses significant obstacles to the accuracy and reliability of the generated summaries. In this paper, we aim to reduce hallucinated outputs or hallucinations in summaries of long-form text documents. We have used the PubMed dataset, which contains long scientific research documents and their abstracts. We have incorporated the techniques of data filtering and joint entity and summary generation (JAENS) in the fine-tuning of the Longformer Encoder-Decoder (LED) model to minimize hallucinations and thereby improve the quality of the generated summary. We have used the following metrics to measure factual consistency at the entity level: precision-source, and F1-target. Our experiments show that the fine-tuned LED model performs well in generating the paper abstract. Data filtering techniques based on some preprocessing steps reduce entity-level hallucinations in the generated summaries in terms of some of the factual consistency metrics.

{{</citation>}}


### (86/151) How many words does ChatGPT know? The answer is ChatWords (Gonzalo Martínez et al., 2023)

{{<citation>}}

Gonzalo Martínez, Javier Conde, Pedro Reviriego, Elena Merino-Gómez, José Alberto Hernández, Fabrizio Lombardi. (2023)  
**How many words does ChatGPT know? The answer is ChatWords**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.16777v1)  

---


**ABSTRACT**  
The introduction of ChatGPT has put Artificial Intelligence (AI) Natural Language Processing (NLP) in the spotlight. ChatGPT adoption has been exponential with millions of users experimenting with it in a myriad of tasks and application domains with impressive results. However, ChatGPT has limitations and suffers hallucinations, for example producing answers that look plausible but they are completely wrong. Evaluating the performance of ChatGPT and similar AI tools is a complex issue that is being explored from different perspectives. In this work, we contribute to those efforts with ChatWords, an automated test system, to evaluate ChatGPT knowledge of an arbitrary set of words. ChatWords is designed to be extensible, easy to use, and adaptable to evaluate also other NLP AI tools. ChatWords is publicly available and its main goal is to facilitate research on the lexical knowledge of AI tools. The benefits of ChatWords are illustrated with two case studies: evaluating the knowledge that ChatGPT has of the Spanish lexicon (taken from the official dictionary of the "Real Academia Espa\~nola") and of the words that appear in the Quixote, the well-known novel written by Miguel de Cervantes. The results show that ChatGPT is only able to recognize approximately 80% of the words in the dictionary and 90% of the words in the Quixote, in some cases with an incorrect meaning. The implications of the lexical knowledge of NLP AI tools and potential applications of ChatWords are also discussed providing directions for further work on the study of the lexical knowledge of AI tools.

{{</citation>}}


### (87/151) Persona-Coded Poly-Encoder: Persona-Guided Multi-Stream Conversational Sentence Scoring (Junfeng Liu et al., 2023)

{{<citation>}}

Junfeng Liu, Christopher Symons, Ranga Raju Vatsavai. (2023)  
**Persona-Coded Poly-Encoder: Persona-Guided Multi-Stream Conversational Sentence Scoring**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, BLEU  
[Paper Link](http://arxiv.org/abs/2309.16770v1)  

---


**ABSTRACT**  
Recent advances in machine learning and deep learning have led to the widespread use of Conversational AI in many practical applications. However, it is still very challenging to leverage auxiliary information that can provide conversational context or personalized tuning to improve the quality of conversations. For example, there has only been limited research on using an individuals persona information to improve conversation quality, and even state-of-the-art conversational AI techniques are unable to effectively leverage signals from heterogeneous sources of auxiliary data, such as multi-modal interaction data, demographics, SDOH data, etc. In this paper, we present a novel Persona-Coded Poly-Encoder method that leverages persona information in a multi-stream encoding scheme to improve the quality of response generation for conversations. To show the efficacy of the proposed method, we evaluate our method on two different persona-based conversational datasets, and compared against two state-of-the-art methods. Our experimental results and analysis demonstrate that our method can improve conversation quality over the baseline method Poly-Encoder by 3.32% and 2.94% in terms of BLEU score and HR@1, respectively. More significantly, our method offers a path to better utilization of multi-modal data in conversational tasks. Lastly, our study outlines several challenges and future research directions for advancing personalized conversational AI technology.

{{</citation>}}


### (88/151) MindShift: Leveraging Large Language Models for Mental-States-Based Problematic Smartphone Use Intervention (Ruolan Wu et al., 2023)

{{<citation>}}

Ruolan Wu, Chun Yu, Xiaole Pan, Yujia Liu, Ningning Zhang, Yue Fu, Yuhan Wang, Zhi Zheng, Li Chen, Qiaolei Jiang, Xuhai Xu, Yuanchun Shi. (2023)  
**MindShift: Leveraging Large Language Models for Mental-States-Based Problematic Smartphone Use Intervention**  

---
Primary Category: cs.CL  
Categories: 68U35, H-5-2; I-2-7, cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16639v1)  

---


**ABSTRACT**  
Problematic smartphone use negatively affects physical and mental health. Despite the wide range of prior research, existing persuasive techniques are not flexible enough to provide dynamic persuasion content based on users' physical contexts and mental states. We first conduct a Wizard-of-Oz study (N=12) and an interview study (N=10) to summarize the mental states behind problematic smartphone use: boredom, stress, and inertia. This informs our design of four persuasion strategies: understanding, comforting, evoking, and scaffolding habits. We leverage large language models (LLMs) to enable the automatic and dynamic generation of effective persuasion content. We develop MindShift, a novel LLM-powered problematic smartphone use intervention technique. MindShift takes users' in-the-moment physical contexts, mental states, app usage behaviors, users' goals & habits as input, and generates high-quality and flexible persuasive content with appropriate persuasion strategies. We conduct a 5-week field experiment (N=25) to compare MindShift with baseline techniques. The results show that MindShift significantly improves intervention acceptance rates by 17.8-22.5% and reduces smartphone use frequency by 12.1-14.4%. Moreover, users have a significant drop in smartphone addiction scale scores and a rise in self-efficacy. Our study sheds light on the potential of leveraging LLMs for context-aware persuasion in other behavior change domains.

{{</citation>}}


### (89/151) Stress Testing Chain-of-Thought Prompting for Large Language Models (Aayush Mishra et al., 2023)

{{<citation>}}

Aayush Mishra, Karan Thakkar. (2023)  
**Stress Testing Chain-of-Thought Prompting for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16621v1)  

---


**ABSTRACT**  
This report examines the effectiveness of Chain-of-Thought (CoT) prompting in improving the multi-step reasoning abilities of large language models (LLMs). Inspired by previous studies \cite{Min2022RethinkingWork}, we analyze the impact of three types of CoT prompt perturbations, namely CoT order, CoT values, and CoT operators on the performance of GPT-3 on various tasks. Our findings show that incorrect CoT prompting leads to poor performance on accuracy metrics. Correct values in the CoT is crucial for predicting correct answers. Moreover, incorrect demonstrations, where the CoT operators or the CoT order are wrong, do not affect the performance as drastically when compared to the value based perturbations. This research deepens our understanding of CoT prompting and opens some new questions regarding the capability of LLMs to learn reasoning in context.

{{</citation>}}


### (90/151) Qwen Technical Report (Jinze Bai et al., 2023)

{{<citation>}}

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, Tianhang Zhu. (2023)  
**Qwen Technical Report**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16609v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce Qwen, the first installment of our large language model series. Qwen is a comprehensive language model series that encompasses distinct models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, Code-Qwen and Code-Qwen-Chat, as well as mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind the proprietary models.

{{</citation>}}


### (91/151) Unlikelihood Tuning on Negative Samples Amazingly Improves Zero-Shot Translation (Changtong Zan et al., 2023)

{{<citation>}}

Changtong Zan, Liang Ding, Li Shen, Yibin Lei, Yibing Zhan, Weifeng Liu, Dacheng Tao. (2023)  
**Unlikelihood Tuning on Negative Samples Amazingly Improves Zero-Shot Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.16599v1)  

---


**ABSTRACT**  
Zero-shot translation (ZST), which is generally based on a multilingual neural machine translation model, aims to translate between unseen language pairs in training data. The common practice to guide the zero-shot language mapping during inference is to deliberately insert the source and target language IDs, e.g., <EN> for English and <DE> for German. Recent studies have shown that language IDs sometimes fail to navigate the ZST task, making them suffer from the off-target problem (non-target language words exist in the generated translation) and, therefore, difficult to apply the current multilingual translation model to a broad range of zero-shot language scenarios. To understand when and why the navigation capabilities of language IDs are weakened, we compare two extreme decoder input cases in the ZST directions: Off-Target (OFF) and On-Target (ON) cases. By contrastively visualizing the contextual word representations (CWRs) of these cases with teacher forcing, we show that 1) the CWRs of different languages are effectively distributed in separate regions when the sentence and ID are matched (ON setting), and 2) if the sentence and ID are unmatched (OFF setting), the CWRs of different languages are chaotically distributed. Our analyses suggest that although they work well in ideal ON settings, language IDs become fragile and lose their navigation ability when faced with off-target tokens, which commonly exist during inference but are rare in training scenarios. In response, we employ unlikelihood tuning on the negative (OFF) samples to minimize their probability such that the language IDs can discriminate between the on- and off-target tokens during training. Experiments spanning 40 ZST directions show that our method reduces the off-target ratio by -48.0% on average, leading to a +9.1 BLEU improvement with only an extra +0.3% tuning cost.

{{</citation>}}


### (92/151) GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond (Shen Zheng et al., 2023)

{{<citation>}}

Shen Zheng, Yuyu Zhang, Yijie Zhu, Chenguang Xi, Pengyang Gao, Xun Zhou, Kevin Chen-Chuan Chang. (2023)  
**GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16583v1)  

---


**ABSTRACT**  
With the rapid advancement of large language models (LLMs), there is a pressing need for a comprehensive evaluation suite to assess their capabilities and limitations. Existing LLM leaderboards often reference scores reported in other papers without consistent settings and prompts, which may inadvertently encourage cherry-picking favored settings and prompts for better results. In this work, we introduce GPT-Fathom, an open-source and reproducible LLM evaluation suite built on top of OpenAI Evals. We systematically evaluate 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. Our retrospective study on OpenAI's earlier models offers valuable insights into the evolutionary path from GPT-3 to GPT-4. Currently, the community is eager to know how GPT-3 progressively improves to GPT-4, including technical details like whether adding code data improves LLM's reasoning capability, which aspects of LLM capability can be improved by SFT and RLHF, how much is the alignment tax, etc. Our analysis sheds light on many of these questions, aiming to improve the transparency of advanced LLMs.

{{</citation>}}


### (93/151) A Benchmark for Learning to Translate a New Language from One Grammar Book (Garrett Tanzer et al., 2023)

{{<citation>}}

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, Luke Melas-Kyriazi. (2023)  
**A Benchmark for Learning to Translate a New Language from One Grammar Book**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.16575v1)  

---


**ABSTRACT**  
Large language models (LLMs) can perform impressive feats with in-context learning or lightweight finetuning. It is natural to wonder how well these models adapt to genuinely new tasks, but how does one find tasks that are unseen in internet-scale training sets? We turn to a field that is explicitly motivated and bottlenecked by a scarcity of web data: low-resource languages. In this paper, we introduce MTOB (Machine Translation from One Book), a benchmark for learning to translate between English and Kalamang -- a language with less than 200 speakers and therefore virtually no presence on the web -- using several hundred pages of field linguistics reference materials. This task framing is novel in that it asks a model to learn a language from a single human-readable book of grammar explanations, rather than a large mined corpus of in-domain data, more akin to L2 learning than L1 acquisition. We demonstrate that baselines using current LLMs are promising but fall short of human performance, achieving 44.7 chrF on Kalamang to English translation and 45.8 chrF on English to Kalamang translation, compared to 51.6 and 57.0 chrF by a human who learned Kalamang from the same reference materials. We hope that MTOB will help measure LLM capabilities along a new dimension, and that the methods developed to solve it could help expand access to language technology for underserved communities by leveraging qualitatively different kinds of data than traditional machine translation.

{{</citation>}}


### (94/151) Unsupervised Fact Verification by Language Model Distillation (Adrián Bazaga et al., 2023)

{{<citation>}}

Adrián Bazaga, Pietro Liò, Gos Micklem. (2023)  
**Unsupervised Fact Verification by Language Model Distillation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: Fact Verification, Language Model, Model Distillation  
[Paper Link](http://arxiv.org/abs/2309.16540v1)  

---


**ABSTRACT**  
Unsupervised fact verification aims to verify a claim using evidence from a trustworthy knowledge base without any kind of data annotation. To address this challenge, algorithms must produce features for every claim that are both semantically meaningful, and compact enough to find a semantic alignment with the source information. In contrast to previous work, which tackled the alignment problem by learning over annotated corpora of claims and their corresponding labels, we propose SFAVEL (Self-supervised Fact Verification via Language Model Distillation), a novel unsupervised framework that leverages pre-trained language models to distil self-supervised features into high-quality claim-fact alignments without the need for annotations. This is enabled by a novel contrastive loss function that encourages features to attain high-quality claim and evidence alignments whilst preserving the semantic relationships across the corpora. Notably, we present results that achieve a new state-of-the-art on the standard FEVER fact verification benchmark (+8% accuracy) with linear evaluation.

{{</citation>}}


### (95/151) KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models (Yiming Ju et al., 2023)

{{<citation>}}

Yiming Ju, Zheng Zhang. (2023)  
**KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16535v1)  

---


**ABSTRACT**  
Recently, Locate-Then-Edit paradigm has emerged as one of the main approaches in changing factual knowledge stored in the Language models. However, there is a lack of research on whether present locating methods can pinpoint the exact parameters embedding the desired knowledge. Moreover, although many researchers have questioned the validity of locality hypothesis of factual knowledge, no method is provided to test the a hypothesis for more in-depth discussion and research. Therefore, we introduce KLoB, a benchmark examining three essential properties that a reliable knowledge locating method should satisfy. KLoB can serve as a benchmark for evaluating existing locating methods in language models, and can contributes a method to reassessing the validity of locality hypothesis of factual knowledge. Our is publicly available at \url{https://github.com/juyiming/KLoB}.

{{</citation>}}


### (96/151) Chatmap : Large Language Model Interaction with Cartographic Data (Eren Unlu, 2023)

{{<citation>}}

Eren Unlu. (2023)  
**Chatmap : Large Language Model Interaction with Cartographic Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01429v1)  

---


**ABSTRACT**  
The swift advancement and widespread availability of foundational Large Language Models (LLMs), complemented by robust fine-tuning methodologies, have catalyzed their adaptation for innovative and industrious applications. Enabling LLMs to recognize and interpret geospatial data, while offering a linguistic access to vast cartographic datasets, is of significant importance. OpenStreetMap (OSM) is the most ambitious open-source global initiative offering detailed urban and rural geographic data, curated by a community of over 10 million contributors, which constitutes a great potential for LLM applications. In this study, we demonstrate the proof of concept and details of the process of fine-tuning a relatively small scale (1B parameters) LLM with a relatively small artificial dataset curated by a more capable teacher model, in order to provide a linguistic interface to the OSM data of an arbitrary urban region. Through this interface, users can inquire about a location's attributes, covering a wide spectrum of concepts, such as its touristic appeal or the potential profitability of various businesses in that vicinity. The study aims to provide an initial guideline for such generative artificial intelligence (AI) adaptations and demonstrate early signs of useful emerging abilities in this context even in minimal computational settings. The embeddings of artificially curated prompts including OSM data are also investigated in detail, which might be instrumental for potential geospatially aware urban Retrieval Augmented Generation (RAG) applications.

{{</citation>}}


### (97/151) Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection (Jiaying Wu et al., 2023)

{{<citation>}}

Jiaying Wu, Shen Li, Ailin Deng, Miao Xiong, Bryan Hooi. (2023)  
**Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SI, cs.CL  
Keywords: Fake News, Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.16424v1)  

---


**ABSTRACT**  
Despite considerable advances in automated fake news detection, due to the timely nature of news, it remains a critical open question how to effectively predict the veracity of news articles based on limited fact-checks. Existing approaches typically follow a "Train-from-Scratch" paradigm, which is fundamentally bounded by the availability of large-scale annotated data. While expressive pre-trained language models (PLMs) have been adapted in a "Pre-Train-and-Fine-Tune" manner, the inconsistency between pre-training and downstream objectives also requires costly task-specific supervision. In this paper, we propose "Prompt-and-Align" (P&A), a novel prompt-based paradigm for few-shot fake news detection that jointly leverages the pre-trained knowledge in PLMs and the social context topology. Our approach mitigates label scarcity by wrapping the news article in a task-related textual prompt, which is then processed by the PLM to directly elicit task-specific knowledge. To supplement the PLM with social context without inducing additional training overheads, motivated by empirical observation on user veracity consistency (i.e., social users tend to consume news of the same veracity type), we further construct a news proximity graph among news articles to capture the veracity-consistent signals in shared readerships, and align the prompting predictions along the graph edges in a confidence-informed manner. Extensive experiments on three real-world benchmarks demonstrate that P&A sets new states-of-the-art for few-shot fake news detection performance by significant margins.

{{</citation>}}


### (98/151) A Comprehensive Survey of Document-level Relation Extraction (2016-2023) (Julien Delaunay et al., 2023)

{{<citation>}}

Julien Delaunay, Thi Hong Hanh Tran, Carlos-Emiliano González-Gallardo, Georgeta Bordea, Nicolas Sidere, Antoine Doucet. (2023)  
**A Comprehensive Survey of Document-level Relation Extraction (2016-2023)**  

---
Primary Category: cs.CL  
Categories: A-1, cs-CL, cs.CL  
Keywords: NLP, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2309.16396v2)  

---


**ABSTRACT**  
Document-level relation extraction (DocRE) is an active area of research in natural language processing (NLP) concerned with identifying and extracting relationships between entities beyond sentence boundaries. Compared to the more traditional sentence-level relation extraction, DocRE provides a broader context for analysis and is more challenging because it involves identifying relationships that may span multiple sentences or paragraphs. This task has gained increased interest as a viable solution to build and populate knowledge bases automatically from unstructured large-scale documents (e.g., scientific papers, legal contracts, or news articles), in order to have a better understanding of relationships between entities. This paper aims to provide a comprehensive overview of recent advances in this field, highlighting its different applications in comparison to sentence-level relation extraction.

{{</citation>}}


### (99/151) Human Feedback is not Gold Standard (Tom Hosking et al., 2023)

{{<citation>}}

Tom Hosking, Phil Blunsom, Max Bartolo. (2023)  
**Human Feedback is not Gold Standard**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16349v1)  

---


**ABSTRACT**  
Human feedback has become the de facto standard for evaluating the performance of Large Language Models, and is increasingly being used as a training objective. However, it is not clear which properties of a generated output this single `preference' score captures. We hypothesise that preference scores are subjective and open to undesirable biases. We critically analyse the use of human feedback for both training and evaluation, to verify whether it fully captures a range of crucial error criteria. We find that while preference scores have fairly good coverage, they under-represent important aspects like factuality. We further hypothesise that both preference scores and error annotation may be affected by confounders, and leverage instruction-tuned models to generate outputs that vary along two possible confounding dimensions: assertiveness and complexity. We find that the assertiveness of an output skews the perceived rate of factuality errors, indicating that human annotations are not a fully reliable evaluation metric or training objective. Finally, we offer preliminary evidence that using human feedback as a training objective disproportionately increases the assertiveness of model outputs. We encourage future work to carefully consider whether preference scores are well aligned with the desired objective.

{{</citation>}}


### (100/151) Augmenting transformers with recursively composed multi-grained representations (Xiang Hu et al., 2023)

{{<citation>}}

Xiang Hu, Qingyang Zhu, Kewei Tu, Wei Wu. (2023)  
**Augmenting transformers with recursively composed multi-grained representations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16319v1)  

---


**ABSTRACT**  
We present ReCAT, a recursive composition augmented Transformer that is able to explicitly model hierarchical syntactic structures of raw texts without relying on gold trees during both learning and inference. Existing research along this line restricts data to follow a hierarchical tree structure and thus lacks inter-span communications. To overcome the problem, we propose a novel contextual inside-outside (CIO) layer that learns contextualized representations of spans through bottom-up and top-down passes, where a bottom-up pass forms representations of high-level spans by composing low-level spans, while a top-down pass combines information inside and outside a span. By stacking several CIO layers between the embedding layer and the attention layers in Transformer, the ReCAT model can perform both deep intra-span and deep inter-span interactions, and thus generate multi-grained representations fully contextualized with other spans. Moreover, the CIO layers can be jointly pre-trained with Transformers, making ReCAT enjoy scaling ability, strong performance, and interpretability at the same time. We conduct experiments on various sentence-level and span-level tasks. Evaluation results indicate that ReCAT can significantly outperform vanilla Transformer models on all span-level tasks and baselines that combine recursive networks with Transformers on natural language inference tasks. More interestingly, the hierarchical structures induced by ReCAT exhibit strong consistency with human-annotated syntactic trees, indicating good interpretability brought by the CIO layers.

{{</citation>}}


### (101/151) At Which Training Stage Does Code Data Help LLMs Reasoning? (Yingwei Ma et al., 2023)

{{<citation>}}

Yingwei Ma, Yue Liu, Yue Yu, Yuanliang Zhang, Yu Jiang, Changjian Wang, Shanshan Li. (2023)  
**At Which Training Stage Does Code Data Help LLMs Reasoning?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.16298v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have exhibited remarkable reasoning capabilities and become the foundation of language technologies. Inspired by the great success of code data in training LLMs, we naturally wonder at which training stage introducing code data can really help LLMs reasoning. To this end, this paper systematically explores the impact of code data on LLMs at different stages. Concretely, we introduce the code data at the pre-training stage, instruction-tuning stage, and both of them, respectively. Then, the reasoning capability of LLMs is comprehensively and fairly evaluated via six reasoning tasks in five domains. We critically analyze the experimental results and provide conclusions with insights. First, pre-training LLMs with the mixture of code and text can significantly enhance LLMs' general reasoning capability almost without negative transfer on other tasks. Besides, at the instruction-tuning stage, code data endows LLMs the task-specific reasoning capability. Moreover, the dynamic mixing strategy of code and text data assists LLMs to learn reasoning capability step-by-step during training. These insights deepen the understanding of LLMs regarding reasoning ability for their application, such as scientific question answering, legal support, etc. The source code and model parameters are released at the link:~\url{https://github.com/yingweima2022/CodeLLM}.

{{</citation>}}


### (102/151) LawBench: Benchmarking Legal Knowledge of Large Language Models (Zhiwei Fei et al., 2023)

{{<citation>}}

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Songyang Zhang, Kai Chen, Zongwen Shen, Jidong Ge. (2023)  
**LawBench: Benchmarking Legal Knowledge of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2309.16289v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated strong capabilities in various aspects. However, when applying them to the highly specialized, safe-critical legal domain, it is unclear how much legal knowledge they possess and whether they can reliably perform legal-related tasks. To address this gap, we propose a comprehensive evaluation benchmark LawBench. LawBench has been meticulously crafted to have precise assessment of the LLMs' legal capabilities from three cognitive levels: (1) Legal knowledge memorization: whether LLMs can memorize needed legal concepts, articles and facts; (2) Legal knowledge understanding: whether LLMs can comprehend entities, events and relationships within legal text; (3) Legal knowledge applying: whether LLMs can properly utilize their legal knowledge and make necessary reasoning steps to solve realistic legal tasks. LawBench contains 20 diverse tasks covering 5 task types: single-label classification (SLC), multi-label classification (MLC), regression, extraction and generation. We perform extensive evaluations of 51 LLMs on LawBench, including 20 multilingual LLMs, 22 Chinese-oriented LLMs and 9 legal specific LLMs. The results show that GPT-4 remains the best-performing LLM in the legal domain, surpassing the others by a significant margin. While fine-tuning LLMs on legal specific text brings certain improvements, we are still a long way from obtaining usable and reliable LLMs in legal tasks. All data, model predictions and evaluation code are released in https://github.com/open-compass/LawBench/. We hope this benchmark provides in-depth understanding of the LLMs' domain-specified capabilities and speed up the development of LLMs in the legal domain.

{{</citation>}}


### (103/151) UPB @ ACTI: Detecting Conspiracies using fine tuned Sentence Transformers (Andrei Paraschiv et al., 2023)

{{<citation>}}

Andrei Paraschiv, Mihai Dascalu. (2023)  
**UPB @ ACTI: Detecting Conspiracies using fine tuned Sentence Transformers**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16275v1)  

---


**ABSTRACT**  
Conspiracy theories have become a prominent and concerning aspect of online discourse, posing challenges to information integrity and societal trust. As such, we address conspiracy theory detection as proposed by the ACTI @ EVALITA 2023 shared task. The combination of pre-trained sentence Transformer models and data augmentation techniques enabled us to secure first place in the final leaderboard of both sub-tasks. Our methodology attained F1 scores of 85.71% in the binary classification and 91.23% for the fine-grained conspiracy topic classification, surpassing other competing systems.

{{</citation>}}


### (104/151) Social Media Fashion Knowledge Extraction as Captioning (Yifei Yuan et al., 2023)

{{<citation>}}

Yifei Yuan, Wenxuan Zhang, Yang Deng, Wai Lam. (2023)  
**Social Media Fashion Knowledge Extraction as Captioning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2309.16270v1)  

---


**ABSTRACT**  
Social media plays a significant role in boosting the fashion industry, where a massive amount of fashion-related posts are generated every day. In order to obtain the rich fashion information from the posts, we study the task of social media fashion knowledge extraction. Fashion knowledge, which typically consists of the occasion, person attributes, and fashion item information, can be effectively represented as a set of tuples. Most previous studies on fashion knowledge extraction are based on the fashion product images without considering the rich text information in social media posts. Existing work on fashion knowledge extraction in social media is classification-based and requires to manually determine a set of fashion knowledge categories in advance. In our work, we propose to cast the task as a captioning problem to capture the interplay of the multimodal post information. Specifically, we transform the fashion knowledge tuples into a natural language caption with a sentence transformation method. Our framework then aims to generate the sentence-based fashion knowledge directly from the social media post. Inspired by the big success of pre-trained models, we build our model based on a multimodal pre-trained generative model and design several auxiliary tasks for enhancing the knowledge extraction. Since there is no existing dataset which can be directly borrowed to our task, we introduce a dataset consisting of social media posts with manual fashion knowledge annotation. Extensive experiments are conducted to demonstrate the effectiveness of our model.

{{</citation>}}


### (105/151) On the Challenges of Fully Incremental Neural Dependency Parsing (Ana Ezquerro et al., 2023)

{{<citation>}}

Ana Ezquerro, Carlos Gómez-Rodríguez, David Vilares. (2023)  
**On the Challenges of Fully Incremental Neural Dependency Parsing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16254v1)  

---


**ABSTRACT**  
Since the popularization of BiLSTMs and Transformer-based bidirectional encoders, state-of-the-art syntactic parsers have lacked incrementality, requiring access to the whole sentence and deviating from human language processing. This paper explores whether fully incremental dependency parsing with modern architectures can be competitive. We build parsers combining strictly left-to-right neural encoders with fully incremental sequence-labeling and transition-based decoders. The results show that fully incremental parsing with modern architectures considerably lags behind bidirectional parsing, noting the challenges of psycholinguistically plausible parsing.

{{</citation>}}


### (106/151) Spider4SPARQL: A Complex Benchmark for Evaluating Knowledge Graph Question Answering Systems (Catherine Kosten et al., 2023)

{{<citation>}}

Catherine Kosten, Philippe Cudré-Mauroux, Kurt Stockinger. (2023)  
**Spider4SPARQL: A Complex Benchmark for Evaluating Knowledge Graph Question Answering Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.16248v1)  

---


**ABSTRACT**  
With the recent spike in the number and availability of Large Language Models (LLMs), it has become increasingly important to provide large and realistic benchmarks for evaluating Knowledge Graph Question Answering (KBQA) systems. So far the majority of benchmarks rely on pattern-based SPARQL query generation approaches. The subsequent natural language (NL) question generation is conducted through crowdsourcing or other automated methods, such as rule-based paraphrasing or NL question templates. Although some of these datasets are of considerable size, their pitfall lies in their pattern-based generation approaches, which do not always generalize well to the vague and linguistically diverse questions asked by humans in real-world contexts.   In this paper, we introduce Spider4SPARQL - a new SPARQL benchmark dataset featuring 9,693 previously existing manually generated NL questions and 4,721 unique, novel, and complex SPARQL queries of varying complexity. In addition to the NL/SPARQL pairs, we also provide their corresponding 166 knowledge graphs and ontologies, which cover 138 different domains. Our complex benchmark enables novel ways of evaluating the strengths and weaknesses of modern KGQA systems. We evaluate the system with state-of-the-art KGQA systems as well as LLMs, which achieve only up to 45\% execution accuracy, demonstrating that Spider4SPARQL is a challenging benchmark for future research.

{{</citation>}}


### (107/151) Analyzing Political Figures in Real-Time: Leveraging YouTube Metadata for Sentiment Analysis (Danendra Athallariq Harya Putra et al., 2023)

{{<citation>}}

Danendra Athallariq Harya Putra, Arief Purnama Muharram. (2023)  
**Analyzing Political Figures in Real-Time: Leveraging YouTube Metadata for Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.16234v1)  

---


**ABSTRACT**  
Sentiment analysis using big data from YouTube videos metadata can be conducted to analyze public opinions on various political figures who represent political parties. This is possible because YouTube has become one of the platforms for people to express themselves, including their opinions on various political figures. The resulting sentiment analysis can be useful for political executives to gain an understanding of public sentiment and develop appropriate and effective political strategies. This study aimed to build a sentiment analysis system leveraging YouTube videos metadata. The sentiment analysis system was built using Apache Kafka, Apache PySpark, and Hadoop for big data handling; TensorFlow for deep learning handling; and FastAPI for deployment on the server. The YouTube videos metadata used in this study is the video description. The sentiment analysis model was built using LSTM algorithm and produces two types of sentiments: positive and negative sentiments. The sentiment analysis results are then visualized in the form a simple web-based dashboard.

{{</citation>}}


### (108/151) Controllable Text Generation with Residual Memory Transformer (Hanqing Zhang et al., 2023)

{{<citation>}}

Hanqing Zhang, Sun Si, Haiming Wu, Dawei Song. (2023)  
**Controllable Text Generation with Residual Memory Transformer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Text Generation, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16231v1)  

---


**ABSTRACT**  
Large-scale Causal Language Models (CLMs), e.g., GPT3 and ChatGPT, have brought great success in text generation. However, it is still an open challenge to control the generation process of CLM while balancing flexibility, control granularity, and generation efficiency. In this paper, we provide a new alternative for controllable text generation (CTG), by designing a non-intrusive, lightweight control plugin to accompany the generation of CLM at arbitrary time steps. The proposed control plugin, namely Residual Memory Transformer (RMT), has an encoder-decoder setup, which can accept any types of control conditions and cooperate with CLM through a residual learning paradigm, to achieve a more flexible, general, and efficient CTG. Extensive experiments are carried out on various control tasks, in the form of both automatic and human evaluations. The results show the superiority of RMT over a range of state-of-the-art approaches, proving the effectiveness and versatility of our approach.

{{</citation>}}


### (109/151) Marathi-English Code-mixed Text Generation (Dhiraj Amin et al., 2023)

{{<citation>}}

Dhiraj Amin, Sharvari Govilkar, Sagar Kulkarni, Yash Shashikant Lalit, Arshi Ajaz Khwaja, Daries Xavier, Sahil Girijashankar Gupta. (2023)  
**Marathi-English Code-mixed Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Text Generation  
[Paper Link](http://arxiv.org/abs/2309.16202v1)  

---


**ABSTRACT**  
Code-mixing, the blending of linguistic elements from distinct languages to form meaningful sentences, is common in multilingual settings, yielding hybrid languages like Hinglish and Minglish. Marathi, India's third most spoken language, often integrates English for precision and formality. Developing code-mixed language systems, like Marathi-English (Minglish), faces resource constraints. This research introduces a Marathi-English code-mixed text generation algorithm, assessed with Code Mixing Index (CMI) and Degree of Code Mixing (DCM) metrics. Across 2987 code-mixed questions, it achieved an average CMI of 0.2 and an average DCM of 7.4, indicating effective and comprehensible code-mixed sentences. These results offer potential for enhanced NLP tools, bridging linguistic gaps in multilingual societies.

{{</citation>}}


### (110/151) Attention Sorting Combats Recency Bias In Long Context Language Models (Alexander Peysakhovich et al., 2023)

{{<citation>}}

Alexander Peysakhovich, Adam Lerer. (2023)  
**Attention Sorting Combats Recency Bias In Long Context Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01427v1)  

---


**ABSTRACT**  
Current language models often fail to incorporate long contexts efficiently during generation. We show that a major contributor to this issue are attention priors that are likely learned during pre-training: relevant information located earlier in context is attended to less on average. Yet even when models fail to use the information from a relevant document in their response, they still pay preferential attention to that document compared to an irrelevant document at the same position. We leverage this fact to introduce ``attention sorting'': perform one step of decoding, sort documents by the attention they receive (highest attention going last), repeat the process, generate the answer with the newly sorted context. We find that attention sorting improves performance of long context models. Our findings highlight some challenges in using off-the-shelf language models for retrieval augmented generation.

{{</citation>}}


### (111/151) Using Weak Supervision and Data Augmentation in Question Answering (Chumki Basu et al., 2023)

{{<citation>}}

Chumki Basu, Himanshu Garg, Allen McIntosh, Sezai Sablak, John R. Wullert II. (2023)  
**Using Weak Supervision and Data Augmentation in Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Augmentation, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.16175v1)  

---


**ABSTRACT**  
The onset of the COVID-19 pandemic accentuated the need for access to biomedical literature to answer timely and disease-specific questions. During the early days of the pandemic, one of the biggest challenges we faced was the lack of peer-reviewed biomedical articles on COVID-19 that could be used to train machine learning models for question answering (QA). In this paper, we explore the roles weak supervision and data augmentation play in training deep neural network QA models. First, we investigate whether labels generated automatically from the structured abstracts of scholarly papers using an information retrieval algorithm, BM25, provide a weak supervision signal to train an extractive QA model. We also curate new QA pairs using information retrieval techniques, guided by the clinicaltrials.gov schema and the structured abstracts of articles, in the absence of annotated data from biomedical domain experts. Furthermore, we explore augmenting the training data of a deep neural network model with linguistic features from external sources such as lexical databases to account for variations in word morphology and meaning. To better utilize our training data, we apply curriculum learning to domain adaptation, fine-tuning our QA model in stages based on characteristics of the QA pairs. We evaluate our methods in the context of QA models at the core of a system to answer questions about COVID-19.

{{</citation>}}


### (112/151) Large Language Model Soft Ideologization via AI-Self-Consciousness (Xiaotian Zhou et al., 2023)

{{<citation>}}

Xiaotian Zhou, Qian Wang, Xiaofeng Wang, Haixu Tang, Xiaozhong Liu. (2023)  
**Large Language Model Soft Ideologization via AI-Self-Consciousness**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16167v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated human-level performance on a vast spectrum of natural language tasks. However, few studies have addressed the LLM threat and vulnerability from an ideology perspective, especially when they are increasingly being deployed in sensitive domains, e.g., elections and education. In this study, we explore the implications of GPT soft ideologization through the use of AI-self-consciousness. By utilizing GPT self-conversations, AI can be granted a vision to "comprehend" the intended ideology, and subsequently generate finetuning data for LLM ideology injection. When compared to traditional government ideology manipulation techniques, such as information censorship, LLM ideologization proves advantageous; it is easy to implement, cost-effective, and powerful, thus brimming with risks.

{{</citation>}}


### (113/151) The Trickle-down Impact of Reward (In-)consistency on RLHF (Lingfeng Shen et al., 2023)

{{<citation>}}

Lingfeng Shen, Sihao Chen, Linfeng Song, Lifeng Jin, Baolin Peng, Haitao Mi, Daniel Khashabi, Dong Yu. (2023)  
**The Trickle-down Impact of Reward (In-)consistency on RLHF**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16155v1)  

---


**ABSTRACT**  
Standard practice within Reinforcement Learning from Human Feedback (RLHF) involves optimizing against a Reward Model (RM), which itself is trained to reflect human preferences for desirable generations. A notable subject that is understudied is the (in-)consistency of RMs -- whether they can recognize the semantic changes to different prompts and appropriately adapt their reward assignments -- and their impact on the downstream RLHF model.   In this paper, we visit a series of research questions relevant to RM inconsistency: (1) How can we measure the consistency of reward models? (2) How consistent are the existing RMs and how can we improve them? (3) In what ways does reward inconsistency influence the chatbots resulting from the RLHF model training?   We propose Contrast Instructions -- a benchmarking strategy for the consistency of RM. Each example in Contrast Instructions features a pair of lexically similar instructions with different ground truth responses. A consistent RM is expected to rank the corresponding instruction and response higher than other combinations. We observe that current RMs trained with the standard ranking objective fail miserably on Contrast Instructions compared to average humans. To show that RM consistency can be improved efficiently without using extra training budget, we propose two techniques ConvexDA and RewardFusion, which enhance reward consistency through extrapolation during the RM training and inference stage, respectively. We show that RLHF models trained with a more consistent RM yield more useful responses, suggesting that reward inconsistency exhibits a trickle-down effect on the downstream RLHF process.

{{</citation>}}


### (114/151) AE-GPT: Using Large Language Models to Extract Adverse Events from Surveillance Reports-A Use Case with Influenza Vaccine Adverse Events (Yiming Li et al., 2023)

{{<citation>}}

Yiming Li, Jianfu Li, Jianping He, Cui Tao. (2023)  
**AE-GPT: Using Large Language Models to Extract Adverse Events from Surveillance Reports-A Use Case with Influenza Vaccine Adverse Events**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16150v1)  

---


**ABSTRACT**  
Though Vaccines are instrumental in global health, mitigating infectious diseases and pandemic outbreaks, they can occasionally lead to adverse events (AEs). Recently, Large Language Models (LLMs) have shown promise in effectively identifying and cataloging AEs within clinical reports. Utilizing data from the Vaccine Adverse Event Reporting System (VAERS) from 1990 to 2016, this study particularly focuses on AEs to evaluate LLMs' capability for AE extraction. A variety of prevalent LLMs, including GPT-2, GPT-3 variants, GPT-4, and Llama 2, were evaluated using Influenza vaccine as a use case. The fine-tuned GPT 3.5 model (AE-GPT) stood out with a 0.704 averaged micro F1 score for strict match and 0.816 for relaxed match. The encouraging performance of the AE-GPT underscores LLMs' potential in processing medical data, indicating a significant stride towards advanced AE detection, thus presumably generalizable to other AE extraction tasks.

{{</citation>}}


### (115/151) The Confidence-Competence Gap in Large Language Models: A Cognitive Study (Aniket Kumar Singh et al., 2023)

{{<citation>}}

Aniket Kumar Singh, Suman Devkota, Bishal Lamichhane, Uttam Dhakal, Chandra Dhakal. (2023)  
**The Confidence-Competence Gap in Large Language Models: A Cognitive Study**  

---
Primary Category: cs.CL  
Categories: ACM-class: I-2-0, cs-CL, cs-CY, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16145v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have acquired ubiquitous attention for their performances across diverse domains. Our study here searches through LLMs' cognitive abilities and confidence dynamics. We dive deep into understanding the alignment between their self-assessed confidence and actual performance. We exploit these models with diverse sets of questionnaires and real-world scenarios and extract how LLMs exhibit confidence in their responses. Our findings reveal intriguing instances where models demonstrate high confidence even when they answer incorrectly. This is reminiscent of the Dunning-Kruger effect observed in human psychology. In contrast, there are cases where models exhibit low confidence with correct answers revealing potential underestimation biases. Our results underscore the need for a deeper understanding of their cognitive processes. By examining the nuances of LLMs' self-assessment mechanism, this investigation provides noteworthy revelations that serve to advance the functionalities and broaden the potential applications of these formidable language models.

{{</citation>}}


### (116/151) Forgetting Private Textual Sequences in Language Models via Leave-One-Out Ensemble (Zhe Liu et al., 2023)

{{<citation>}}

Zhe Liu, Ozlem Kalinli. (2023)  
**Forgetting Private Textual Sequences in Language Models via Leave-One-Out Ensemble**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16082v1)  

---


**ABSTRACT**  
Recent research has shown that language models have a tendency to memorize rare or unique token sequences in the training corpus. After deploying a model, practitioners might be asked to delete any personal information from the model by individuals' requests. Re-training the underlying model every time individuals would like to practice their rights to be forgotten is computationally expensive. We employ a teacher-student framework and propose a novel leave-one-out ensemble method to unlearn the targeted textual sequences that need to be forgotten from the model. In our approach, multiple teachers are trained on disjoint sets; for each targeted sequence to be removed, we exclude the teacher trained on the set containing this sequence and aggregate the predictions from remaining teachers to provide supervision during fine-tuning. Experiments on LibriSpeech and WikiText-103 datasets show that the proposed method achieves superior privacy-utility trade-offs than other counterparts.

{{</citation>}}


## cs.CR (2)



### (117/151) Layered Security Guidance for Data Asset Management in Additive Manufacturing (Fahad Ali Milaat et al., 2023)

{{<citation>}}

Fahad Ali Milaat, Joshua Lubell. (2023)  
**Layered Security Guidance for Data Asset Management in Additive Manufacturing**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.16842v1)  

---


**ABSTRACT**  
Manufacturing industries are increasingly adopting additive manufacturing (AM) technologies to produce functional parts in critical systems. However, the inherent complexity of both AM designs and AM processes render them attractive targets for cyber-attacks. Risk-based Information Technology (IT) and Operational Technology (OT) security guidance standards are useful resources for AM security practitioners, but the guidelines they provide are insufficient without additional AM-specific revisions. Therefore, a structured layering approach is needed to efficiently integrate these revisions with preexisting IT and OT security guidance standards. To implement such an approach, this paper proposes leveraging the National Institute of Standards and Technology's Cybersecurity Framework (CSF) to develop layered, risk-based guidance for fulfilling specific security outcomes. It begins with an in-depth literature review that reveals the importance of AM data and asset management to risk-based security. Next, this paper adopts the CSF asset identification and management security outcomes as an example for providing AM-specific guidance and identifies the AM geometry and process definitions to aid manufacturers in mapping data flows and documenting processes. Finally, this paper uses the Open Security Controls Assessment Language to integrate the AM-specific guidance together with existing IT and OT security guidance in a rigorous and traceable manner. This paper's contribution is to show how a risk-based layered approach enables the authoring, publishing, and management of AM-specific security guidance that is currently lacking. The authors believe implementation of the layered approach would result in value-added, non-redundant security guidance for AM that is consistent with the preexisting guidance.

{{</citation>}}


### (118/151) Cyber Sentinel: Exploring Conversational Agents in Streamlining Security Tasks with GPT-4 (Mehrdad Kaheh et al., 2023)

{{<citation>}}

Mehrdad Kaheh, Danial Khosh Kholgh, Panos Kostakos. (2023)  
**Cyber Sentinel: Exploring Conversational Agents in Streamlining Security Tasks with GPT-4**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, GPT, GPT-4, Security  
[Paper Link](http://arxiv.org/abs/2309.16422v1)  

---


**ABSTRACT**  
In an era where cyberspace is both a battleground and a backbone of modern society, the urgency of safeguarding digital assets against ever-evolving threats is paramount. This paper introduces Cyber Sentinel, an innovative task-oriented cybersecurity dialogue system that is effectively capable of managing two core functions: explaining potential cyber threats within an organization to the user, and taking proactive/reactive security actions when instructed by the user. Cyber Sentinel embodies the fusion of artificial intelligence, cybersecurity domain expertise, and real-time data analysis to combat the multifaceted challenges posed by cyber adversaries. This article delves into the process of creating such a system and how it can interact with other components typically found in cybersecurity organizations. Our work is a novel approach to task-oriented dialogue systems, leveraging the power of chaining GPT-4 models combined with prompt engineering across all sub-tasks. We also highlight its pivotal role in enhancing cybersecurity communication and interaction, concluding that not only does this framework enhance the system's transparency (Explainable AI) but also streamlines the decision-making process and responding to threats (Actionable AI), therefore marking a significant advancement in the realm of cybersecurity communication.

{{</citation>}}


## eess.IV (2)



### (119/151) Class Activation Map-based Weakly supervised Hemorrhage Segmentation using Resnet-LSTM in Non-Contrast Computed Tomography images (Shreyas H Ramananda et al., 2023)

{{<citation>}}

Shreyas H Ramananda, Vaanathi Sundaresan. (2023)  
**Class Activation Map-based Weakly supervised Hemorrhage Segmentation using Resnet-LSTM in Non-Contrast Computed Tomography images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2309.16627v1)  

---


**ABSTRACT**  
In clinical settings, intracranial hemorrhages (ICH) are routinely diagnosed using non-contrast CT (NCCT) for severity assessment. Accurate automated segmentation of ICH lesions is the initial and essential step, immensely useful for such assessment. However, compared to other structural imaging modalities such as MRI, in NCCT images ICH appears with very low contrast and poor SNR. Over recent years, deep learning (DL)-based methods have shown great potential, however, training them requires a huge amount of manually annotated lesion-level labels, with sufficient diversity to capture the characteristics of ICH. In this work, we propose a novel weakly supervised DL method for ICH segmentation on NCCT scans, using image-level binary classification labels, which are less time-consuming and labor-efficient when compared to the manual labeling of individual ICH lesions. Our method initially determines the approximate location of ICH using class activation maps from a classification network, which is trained to learn dependencies across contiguous slices. We further refine the ICH segmentation using pseudo-ICH masks obtained in an unsupervised manner. The method is flexible and uses a computationally light architecture during testing. On evaluating our method on the validation data of the MICCAI 2022 INSTANCE challenge, our method achieves a Dice value of 0.55, comparable with those of existing weakly supervised method (Dice value of 0.47), despite training on a much smaller training data.

{{</citation>}}


### (120/151) Cross-Modal Transformer GAN: Brain Structural-Functional Deep Fusing Network for Alzheimer's Disease Analysis (Qiankun Zuo et al., 2023)

{{<citation>}}

Qiankun Zuo, Junren Pan, Shuqiang Wang. (2023)  
**Cross-Modal Transformer GAN: Brain Structural-Functional Deep Fusing Network for Alzheimer's Disease Analysis**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.16206v1)  

---


**ABSTRACT**  
Fusing structural-functional images of the brain has shown great potential to analyze the deterioration of Alzheimer's disease (AD). However, it is a big challenge to effectively fuse the correlated and complementary information from multimodal neuroimages. In this paper, a novel model termed cross-modal transformer generative adversarial network (CT-GAN) is proposed to effectively fuse the functional and structural information contained in functional magnetic resonance imaging (fMRI) and diffusion tensor imaging (DTI). The CT-GAN can learn topological features and generate multimodal connectivity from multimodal imaging data in an efficient end-to-end manner. Moreover, the swapping bi-attention mechanism is designed to gradually align common features and effectively enhance the complementary features between modalities. By analyzing the generated connectivity features, the proposed model can identify AD-related brain connections. Evaluations on the public ADNI dataset show that the proposed CT-GAN can dramatically improve prediction performance and detect AD-related brain regions effectively. The proposed model also provides new insights for detecting AD-related abnormal neural circuits.

{{</citation>}}


## stat.ML (2)



### (121/151) Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit (Blake Bordelon et al., 2023)

{{<citation>}}

Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, Cengiz Pehlevan. (2023)  
**Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit**  

---
Primary Category: stat.ML  
Categories: cond-mat-dis-nn, cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16620v1)  

---


**ABSTRACT**  
The cost of hyperparameter tuning in deep learning has been rising with model sizes, prompting practitioners to find new tuning methods using a proxy of smaller networks. One such proposal uses $\mu$P parameterized networks, where the optimal hyperparameters for small width networks transfer to networks with arbitrarily large width. However, in this scheme, hyperparameters do not transfer across depths. As a remedy, we study residual networks with a residual branch scale of $1/\sqrt{\text{depth}}$ in combination with the $\mu$P parameterization. We provide experiments demonstrating that residual architectures including convolutional ResNets and Vision Transformers trained with this parameterization exhibit transfer of optimal hyperparameters across width and depth on CIFAR-10 and ImageNet. Furthermore, our empirical findings are supported and motivated by theory. Using recent developments in the dynamical mean field theory (DMFT) description of neural network learning dynamics, we show that this parameterization of ResNets admits a well-defined feature learning joint infinite-width and infinite-depth limit and show convergence of finite-size network dynamics towards this limit.

{{</citation>}}


### (122/151) Generating Personalized Insulin Treatments Strategies with Deep Conditional Generative Time Series Models (Manuel Schürch et al., 2023)

{{<citation>}}

Manuel Schürch, Xiang Li, Ahmed Allam, Giulia Rathmes, Amina Mollaysa, Claudia Cavelti-Weder, Michael Krauthammer. (2023)  
**Generating Personalized Insulin Treatments Strategies with Deep Conditional Generative Time Series Models**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.16521v1)  

---


**ABSTRACT**  
We propose a novel framework that combines deep generative time series models with decision theory for generating personalized treatment strategies. It leverages historical patient trajectory data to jointly learn the generation of realistic personalized treatment and future outcome trajectories through deep generative time series models. In particular, our framework enables the generation of novel multivariate treatment strategies tailored to the personalized patient history and trained for optimal expected future outcomes based on conditional expected utility maximization. We demonstrate our framework by generating personalized insulin treatment strategies and blood glucose predictions for hospitalized diabetes patients, showcasing the potential of our approach for generating improved personalized treatment strategies. Keywords: deep generative model, probabilistic decision support, personalized treatment generation, insulin and blood glucose prediction

{{</citation>}}


## cs.HC (3)



### (123/151) 'AI enhances our performance, I have no doubt this one will do the same': The Placebo effect is robust to negative descriptions of AI (Agnes M. Kloft et al., 2023)

{{<citation>}}

Agnes M. Kloft, Robin Welsch, Thomas Kosch, Steeven Villa. (2023)  
**'AI enhances our performance, I have no doubt this one will do the same': The Placebo effect is robust to negative descriptions of AI**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16606v1)  

---


**ABSTRACT**  
Heightened AI expectations facilitate performance in human-AI interactions through placebo effects. While lowering expectations to control for placebo effects is advisable, overly negative expectations could induce nocebo effects. In a letter discrimination task, we informed participants that an AI would either increase or decrease their performance by adapting the interface, but in reality, no AI was present in any condition. A Bayesian analysis showed that participants had high expectations and performed descriptively better irrespective of the AI description when a sham-AI was present. Using cognitive modeling, we could trace this advantage back to participants gathering more information. A replication study verified that negative AI descriptions do not alter expectations, suggesting that performance expectations with AI are biased and robust to negative verbal descriptions. We discuss the impact of user expectations on AI interactions and evaluation and provide a behavioral placebo marker for human-AI interaction

{{</citation>}}


### (124/151) Does Explanation Matter? An Exploratory Study on the Effects of Covid 19 Misinformation Warning Flags on Social Media (Dipto Barman et al., 2023)

{{<citation>}}

Dipto Barman, Owen Conlan. (2023)  
**Does Explanation Matter? An Exploratory Study on the Effects of Covid 19 Misinformation Warning Flags on Social Media**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-SI, cs.HC, physics-soc-ph  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2309.16305v1)  

---


**ABSTRACT**  
We investigate whether adding specific explanations from fact checking websites enhances trust in these flags. We experimented with 348 American participants, exposing them to a randomised order of true and false news headlines related to COVID 19, with and without warning flags and explanation text. Our findings suggest that warning flags, whether alone or accompanied by explanatory text, effectively reduce the perceived accuracy of fake news and the intent to share such headlines. Interestingly, our study also suggests that incorporating explanatory text in misinformation warning systems could significantly enhance their trustworthiness, emphasising the importance of transparency and user comprehension in combating fake news on social media.

{{</citation>}}


### (125/151) ACT2G: Attention-based Contrastive Learning for Text-to-Gesture Generation (Hitoshi Teshima et al., 2023)

{{<citation>}}

Hitoshi Teshima, Naoki Wake, Diego Thomas, Yuta Nakashima, Hiroshi Kawasaki, Katsushi Ikeuchi. (2023)  
**ACT2G: Attention-based Contrastive Learning for Text-to-Gesture Generation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Attention, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.16162v1)  

---


**ABSTRACT**  
Recent increase of remote-work, online meeting and tele-operation task makes people find that gesture for avatars and communication robots is more important than we have thought. It is one of the key factors to achieve smooth and natural communication between humans and AI systems and has been intensively researched. Current gesture generation methods are mostly based on deep neural network using text, audio and other information as the input, however, they generate gestures mainly based on audio, which is called a beat gesture. Although the ratio of the beat gesture is more than 70% of actual human gestures, content based gestures sometimes play an important role to make avatars more realistic and human-like. In this paper, we propose a attention-based contrastive learning for text-to-gesture (ACT2G), where generated gestures represent content of the text by estimating attention weight for each word from the input text. In the method, since text and gesture features calculated by the attention weight are mapped to the same latent space by contrastive learning, once text is given as input, the network outputs a feature vector which can be used to generate gestures related to the content. User study confirmed that the gestures generated by ACT2G were better than existing methods. In addition, it was demonstrated that wide variation of gestures were generated from the same text by changing attention weights by creators.

{{</citation>}}


## cs.AI (7)



### (126/151) Navigating Healthcare Insights: A Birds Eye View of Explainability with Knowledge Graphs (Satvik Garg et al., 2023)

{{<citation>}}

Satvik Garg, Shivam Parikh, Somya Garg. (2023)  
**Navigating Healthcare Insights: A Birds Eye View of Explainability with Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.16593v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) are gaining prominence in Healthcare AI, especially in drug discovery and pharmaceutical research as they provide a structured way to integrate diverse information sources, enhancing AI system interpretability. This interpretability is crucial in healthcare, where trust and transparency matter, and eXplainable AI (XAI) supports decision making for healthcare professionals. This overview summarizes recent literature on the impact of KGs in healthcare and their role in developing explainable AI models. We cover KG workflow, including construction, relationship extraction, reasoning, and their applications in areas like Drug-Drug Interactions (DDI), Drug Target Interactions (DTI), Drug Development (DD), Adverse Drug Reactions (ADR), and bioinformatics. We emphasize the importance of making KGs more interpretable through knowledge-infused learning in healthcare. Finally, we highlight research challenges and provide insights for future directions.

{{</citation>}}


### (127/151) Neuro Symbolic Reasoning for Planning: Counterexample Guided Inductive Synthesis using Large Language Models and Satisfiability Solving (Sumit Kumar Jha et al., 2023)

{{<citation>}}

Sumit Kumar Jha, Susmit Jha, Patrick Lincoln, Nathaniel D. Bastian, Alvaro Velasquez, Rickard Ewetz, Sandeep Neema. (2023)  
**Neuro Symbolic Reasoning for Planning: Counterexample Guided Inductive Synthesis using Large Language Models and Satisfiability Solving**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LO, cs.AI  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.16436v1)  

---


**ABSTRACT**  
Generative large language models (LLMs) with instruct training such as GPT-4 can follow human-provided instruction prompts and generate human-like responses to these prompts. Apart from natural language responses, they have also been found to be effective at generating formal artifacts such as code, plans, and logical specifications from natural language prompts. Despite their remarkably improved accuracy, these models are still known to produce factually incorrect or contextually inappropriate results despite their syntactic coherence - a phenomenon often referred to as hallucination. This limitation makes it difficult to use these models to synthesize formal artifacts that are used in safety-critical applications. Unlike tasks such as text summarization and question-answering, bugs in code, plan, and other formal artifacts produced by LLMs can be catastrophic. We posit that we can use the satisfiability modulo theory (SMT) solvers as deductive reasoning engines to analyze the generated solutions from the LLMs, produce counterexamples when the solutions are incorrect, and provide that feedback to the LLMs exploiting the dialog capability of instruct-trained LLMs. This interaction between inductive LLMs and deductive SMT solvers can iteratively steer the LLM to generate the correct response. In our experiments, we use planning over the domain of blocks as our synthesis task for evaluating our approach. We use GPT-4, GPT3.5 Turbo, Davinci, Curie, Babbage, and Ada as the LLMs and Z3 as the SMT solver. Our method allows the user to communicate the planning problem in natural language; even the formulation of queries to SMT solvers is automatically generated from natural language. Thus, the proposed technique can enable non-expert users to describe their problems in natural language, and the combination of LLMs and SMT solvers can produce provably correct solutions.

{{</citation>}}


### (128/151) RLLTE: Long-Term Evolution Project of Reinforcement Learning (Mingqi Yuan et al., 2023)

{{<citation>}}

Mingqi Yuan, Zequn Zhang, Yang Xu, Shihao Luo, Bo Li, Xin Jin, Wenjun Zeng. (2023)  
**RLLTE: Long-Term Evolution Project of Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16382v1)  

---


**ABSTRACT**  
We present RLLTE: a long-term evolution, extremely modular, and open-source framework for reinforcement learning (RL) research and application. Beyond delivering top-notch algorithm implementations, RLLTE also serves as a toolkit for developing algorithms. More specifically, RLLTE decouples the RL algorithms completely from the exploitation-exploration perspective, providing a large number of components to accelerate algorithm development and evolution. In particular, RLLTE is the first RL framework to build a complete and luxuriant ecosystem, which includes model training, evaluation, deployment, benchmark hub, and large language model (LLM)-empowered copilot. RLLTE is expected to set standards for RL engineering practice and be highly stimulative for industry and academia.

{{</citation>}}


### (129/151) GInX-Eval: Towards In-Distribution Evaluation of Graph Neural Network Explanations (Kenza Amara et al., 2023)

{{<citation>}}

Kenza Amara, Mennatallah El-Assady, Rex Ying. (2023)  
**GInX-Eval: Towards In-Distribution Evaluation of Graph Neural Network Explanations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.16223v1)  

---


**ABSTRACT**  
Diverse explainability methods of graph neural networks (GNN) have recently been developed to highlight the edges and nodes in the graph that contribute the most to the model predictions. However, it is not clear yet how to evaluate the correctness of those explanations, whether it is from a human or a model perspective. One unaddressed bottleneck in the current evaluation procedure is the problem of out-of-distribution explanations, whose distribution differs from those of the training data. This important issue affects existing evaluation metrics such as the popular faithfulness or fidelity score. In this paper, we show the limitations of faithfulness metrics. We propose GInX-Eval (Graph In-distribution eXplanation Evaluation), an evaluation procedure of graph explanations that overcomes the pitfalls of faithfulness and offers new insights on explainability methods. Using a retraining strategy, the GInX score measures how informative removed edges are for the model and the EdgeRank score evaluates if explanatory edges are correctly ordered by their importance. GInX-Eval verifies if ground-truth explanations are instructive to the GNN model. In addition, it shows that many popular methods, including gradient-based methods, produce explanations that are not better than a random designation of edges as important subgraphs, challenging the findings of current works in the area. Results with GInX-Eval are consistent across multiple datasets and align with human evaluation.

{{</citation>}}


### (130/151) CoinRun: Solving Goal Misgeneralisation (Stuart Armstrong et al., 2023)

{{<citation>}}

Stuart Armstrong, Alexandre Maranhão, Oliver Daniels-Koch, Patrick Leask, Rebecca Gorman. (2023)  
**CoinRun: Solving Goal Misgeneralisation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16166v1)  

---


**ABSTRACT**  
Goal misgeneralisation is a key challenge in AI alignment -- the task of getting powerful Artificial Intelligences to align their goals with human intentions and human morality. In this paper, we show how the ACE (Algorithm for Concept Extrapolation) agent can solve one of the key standard challenges in goal misgeneralisation: the CoinRun challenge. It uses no new reward information in the new environment. This points to how autonomous agents could be trusted to act in human interests, even in novel and critical situations.

{{</citation>}}


### (131/151) T-COL: Generating Counterfactual Explanations for General User Preferences on Variable Machine Learning Systems (Ming Wang et al., 2023)

{{<citation>}}

Ming Wang, Daling Wang, Wenfang Wu, Shi Feng, Yifei Zhang. (2023)  
**T-COL: Generating Counterfactual Explanations for General User Preferences on Variable Machine Learning Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.16146v1)  

---


**ABSTRACT**  
Machine learning (ML) based systems have been suffering a lack of interpretability. To address this problem, counterfactual explanations (CEs) have been proposed. CEs are unique as they provide workable suggestions to users, in addition to explaining why a certain outcome was predicted. However, the application of CEs has been hindered by two main challenges, namely general user preferences and variable ML systems. User preferences, in particular, tend to be general rather than specific feature values. Additionally, CEs need to be customized to suit the variability of ML models, while also maintaining robustness even when these validation models change. To overcome these challenges, we propose several possible general user preferences that have been validated by user research and map them to the properties of CEs. We also introduce a new method called \uline{T}ree-based \uline{C}onditions \uline{O}ptional \uline{L}inks (T-COL), which has two optional structures and several groups of conditions for generating CEs that can be adapted to general user preferences. Meanwhile, a group of conditions lead T-COL to generate more robust CEs that have higher validity when the ML model is replaced. We compared the properties of CEs generated by T-COL experimentally under different user preferences and demonstrated that T-COL is better suited for accommodating user preferences and variable ML systems compared to baseline methods including Large Language Models.

{{</citation>}}


### (132/151) TPE: Towards Better Compositional Reasoning over Conceptual Tools with Multi-persona Collaboration (Hongru Wang et al., 2023)

{{<citation>}}

Hongru Wang, Huimin Wang, Lingzhi Wang, Minda Hu, Rui Wang, Boyang Xue, Hongyuan Lu, Fei Mi, Kam-Fai Wong. (2023)  
**TPE: Towards Better Compositional Reasoning over Conceptual Tools with Multi-persona Collaboration**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.16090v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated exceptional performance in planning the use of various functional tools, such as calculators and retrievers, particularly in question-answering tasks. In this paper, we expand the definition of these tools, centering on conceptual tools within the context of dialogue systems. A conceptual tool specifies a cognitive concept that aids systematic or investigative thought. These conceptual tools play important roles in practice, such as multiple psychological or tutoring strategies being dynamically applied in a single turn to compose helpful responses. To further enhance the reasoning and planning capability of LLMs with these conceptual tools, we introduce a multi-persona collaboration framework: Think-Plan-Execute (TPE). This framework decouples the response generation process into three distinct roles: Thinker, Planner, and Executor. Specifically, the Thinker analyzes the internal status exhibited in the dialogue context, such as user emotions and preferences, to formulate a global guideline. The Planner then generates executable plans to call different conceptual tools (e.g., sources or strategies), while the Executor compiles all intermediate results into a coherent response. This structured approach not only enhances the explainability and controllability of responses but also reduces token redundancy. We demonstrate the effectiveness of TPE across various dialogue response generation tasks, including multi-source (FoCus) and multi-strategy interactions (CIMA and PsyQA). This reveals its potential to handle real-world dialogue interactions that require more complicated tool learning beyond just functional tools. The full code and data will be released for reproduction.

{{</citation>}}


## cs.SD (4)



### (133/151) Audio-Visual Speaker Verification via Joint Cross-Attention (R. Gnana Praveen et al., 2023)

{{<citation>}}

R. Gnana Praveen, Jahangir Alam. (2023)  
**Audio-Visual Speaker Verification via Joint Cross-Attention**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Attention, Speaker Verification  
[Paper Link](http://arxiv.org/abs/2309.16569v1)  

---


**ABSTRACT**  
Speaker verification has been widely explored using speech signals, which has shown significant improvement using deep models. Recently, there has been a surge in exploring faces and voices as they can offer more complementary and comprehensive information than relying only on a single modality of speech signals. Though current methods in the literature on the fusion of faces and voices have shown improvement over that of individual face or voice modalities, the potential of audio-visual fusion is not fully explored for speaker verification. Most of the existing methods based on audio-visual fusion either rely on score-level fusion or simple feature concatenation. In this work, we have explored cross-modal joint attention to fully leverage the inter-modal complementary information and the intra-modal information for speaker verification. Specifically, we estimate the cross-attention weights based on the correlation between the joint feature presentation and that of the individual feature representations in order to effectively capture both intra-modal as well inter-modal relationships among the faces and voices. We have shown that efficiently leveraging the intra- and inter-modal relationships significantly improves the performance of audio-visual fusion for speaker verification. The performance of the proposed approach has been evaluated on the Voxceleb1 dataset. Results show that the proposed approach can significantly outperform the state-of-the-art methods of audio-visual fusion for speaker verification.

{{</citation>}}


### (134/151) Efficient Supervised Training of Audio Transformers for Music Representation Learning (Pablo Alonso-Jiménez et al., 2023)

{{<citation>}}

Pablo Alonso-Jiménez, Xavier Serra, Dmitry Bogdanov. (2023)  
**Efficient Supervised Training of Audio Transformers for Music Representation Learning**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: ImageNet, Representation Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16418v1)  

---


**ABSTRACT**  
In this work, we address music representation learning using convolution-free transformers. We build on top of existing spectrogram-based audio transformers such as AST and train our models on a supervised task using patchout training similar to PaSST. In contrast to previous works, we study how specific design decisions affect downstream music tagging tasks instead of focusing on the training task. We assess the impact of initializing the models with different pre-trained weights, using various input audio segment lengths, using learned representations from different blocks and tokens of the transformer for downstream tasks, and applying patchout at inference to speed up feature extraction. We find that 1) initializing the model from ImageNet or AudioSet weights and using longer input segments are beneficial both for the training and downstream tasks, 2) the best representations for the considered downstream tasks are located in the middle blocks of the transformer, and 3) using patchout at inference allows faster processing than our convolutional baselines while maintaining superior performance. The resulting models, MAEST, are publicly available and obtain the best performance among open models in music tagging tasks.

{{</citation>}}


### (135/151) Predicting performance difficulty from piano sheet music images (Pedro Ramoneda et al., 2023)

{{<citation>}}

Pedro Ramoneda, Jose J. Valero-Mas, Dasaem Jeong, Xavier Serra. (2023)  
**Predicting performance difficulty from piano sheet music images**  

---
Primary Category: cs.SD  
Categories: cs-DL, cs-SD, cs.SD, eess-AS  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2309.16287v1)  

---


**ABSTRACT**  
Estimating the performance difficulty of a musical score is crucial in music education for adequately designing the learning curriculum of the students. Although the Music Information Retrieval community has recently shown interest in this task, existing approaches mainly use machine-readable scores, leaving the broader case of sheet music images unaddressed. Based on previous works involving sheet music images, we use a mid-level representation, bootleg score, describing notehead positions relative to staff lines coupled with a transformer model. This architecture is adapted to our task by introducing an encoding scheme that reduces the encoded sequence length to one-eighth of the original size. In terms of evaluation, we consider five datasets -- more than 7500 scores with up to 9 difficulty levels -- , two of them particularly compiled for this work. The results obtained when pretraining the scheme on the IMSLP corpus and fine-tuning it on the considered datasets prove the proposal's validity, achieving the best-performing model with a balanced accuracy of 40.34\% and a mean square error of 1.33. Finally, we provide access to our code, data, and models for transparency and reproducibility.

{{</citation>}}


### (136/151) NOMAD: Unsupervised Learning of Perceptual Embeddings for Speech Enhancement and Non-matching Reference Audio Quality Assessment (Alessandro Ragano et al., 2023)

{{<citation>}}

Alessandro Ragano, Jan Skoglund, Andrew Hines. (2023)  
**NOMAD: Unsupervised Learning of Perceptual Embeddings for Speech Enhancement and Non-matching Reference Audio Quality Assessment**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.16284v1)  

---


**ABSTRACT**  
This paper presents NOMAD (Non-Matching Audio Distance), a differentiable perceptual similarity metric that measures the distance of a degraded signal against non-matching references. The proposed method is based on learning deep feature embeddings via a triplet loss guided by the Neurogram Similarity Index Measure (NSIM) to capture degradation intensity. During inference, the similarity score between any two audio samples is computed through Euclidean distance of their embeddings. NOMAD is fully unsupervised and can be used in general perceptual audio tasks for audio analysis e.g. quality assessment and generative tasks such as speech enhancement and speech synthesis. The proposed method is evaluated with 3 tasks. Ranking degradation intensity, predicting speech quality, and as a loss function for speech enhancement. Results indicate NOMAD outperforms other non-matching reference approaches in both ranking degradation intensity and quality assessment, exhibiting competitive performance with full-reference audio metrics. NOMAD demonstrates a promising technique that mimics human capabilities in assessing audio quality with non-matching references to learn perceptual embeddings without the need for human-generated labels.

{{</citation>}}


## cs.DC (1)



### (137/151) SIMD Everywhere Optimization from ARM NEON to RISC-V Vector Extensions (Ju-Hung Li et al., 2023)

{{<citation>}}

Ju-Hung Li, Jhih-Kuan Lin, Yung-Cheng Su, Chi-Wei Chu, Lai-Tak Kuok, Hung-Ming Lai, Chao-Lin Lee, Jenq-Kuen Lee. (2023)  
**SIMD Everywhere Optimization from ARM NEON to RISC-V Vector Extensions**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-PL, cs.DC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.16509v1)  

---


**ABSTRACT**  
Many libraries, such as OpenCV, FFmpeg, XNNPACK, and Eigen, utilize Arm or x86 SIMD Intrinsics to optimize programs for performance. With the emergence of RISC-V Vector Extensions (RVV), there is a need to migrate these performance legacy codes for RVV. Currently, the migration of NEON code to RVV code requires manual rewriting, which is a time-consuming and error-prone process. In this work, we use the open source tool, "SIMD Everywhere" (SIMDe), to automate the migration. Our primary task is to enhance SIMDe to enable the conversion of ARM NEON Intrinsics types and functions to their corresponding RVV Intrinsics types and functions. For type conversion, we devise strategies to convert Neon Intrinsics types to RVV Intrinsics by considering the vector length agnostic (vla) architectures. With function conversions, we analyze commonly used conversion methods in SIMDe and develop customized conversions for each function based on the results of RVV code generations. In our experiments with Google XNNPACK library, our enhanced SIMDe achieves speedup ranging from 1.51x to 5.13x compared to the original SIMDe, which does not utilize customized RVV implementations for the conversions.

{{</citation>}}


## cs.SI (1)



### (138/151) A Small World of Bad Guys: Investigating the Behavior of Hacker Groups in Cyber-Attacks (Giampiero Giacomello et al., 2023)

{{<citation>}}

Giampiero Giacomello, Antonio Iovanella, Luigi Martino. (2023)  
**A Small World of Bad Guys: Investigating the Behavior of Hacker Groups in Cyber-Attacks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2309.16442v1)  

---


**ABSTRACT**  
This paper explores the behaviour of malicious hacker groups operating in cyberspace and how they organize themselves in structured networks. To better understand these groups, the paper uses Social Network Analysis (SNA) to analyse the interactions and relationships among several malicious hacker groups. The study uses a tested dataset as its primary source, providing an empirical analysis of the cooperative behaviours exhibited by these groups. The study found that malicious hacker groups tend to form close-knit networks where they consult, coordinate with, and assist each other in carrying out their attacks. The study also identified a "small world" phenomenon within the population of malicious actors, which suggests that these groups establish interconnected relationships to facilitate their malicious operations. The small world phenomenon indicates that the actor-groups are densely connected, but they also have a small number of connections to other groups, allowing for efficient communication and coordination of their activities.

{{</citation>}}


## q-fin.GN (1)



### (139/151) Assessing the Solvency of Virtual Asset Service Providers: Are Current Standards Sufficient? (Pietro Saggese et al., 2023)

{{<citation>}}

Pietro Saggese, Esther Segalla, Michael Sigmund, Burkhard Raunig, Felix Zangerl, Bernhard Haslhofer. (2023)  
**Assessing the Solvency of Virtual Asset Service Providers: Are Current Standards Sufficient?**  

---
Primary Category: q-fin.GN  
Categories: cs-CR, q-fin-GN, q-fin.GN  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2309.16408v1)  

---


**ABSTRACT**  
Entities like centralized cryptocurrency exchanges fall under the business category of virtual asset service providers (VASPs). As any other enterprise, they can become insolvent. VASPs enable the exchange, custody, and transfer of cryptoassets organized in wallets across distributed ledger technologies (DLTs). Despite the public availability of DLT transactions, the cryptoasset holdings of VASPs are not yet subject to systematic auditing procedures. In this paper, we propose an approach to assess the solvency of a VASP by cross-referencing data from three distinct sources: cryptoasset wallets, balance sheets from the commercial register, and data from supervisory entities. We investigate 24 VASPs registered with the Financial Market Authority in Austria and provide regulatory data insights such as who are the customers and where do they come from. Their yearly incoming and outgoing transaction volume amount to 2 billion EUR for around 1.8 million users. We describe what financial services they provide and find that they are most similar to traditional intermediaries such as brokers, money exchanges, and funds, rather than banks. Next, we empirically measure DLT transaction flows of four VASPs and compare their cryptoasset holdings to balance sheet entries. Data are consistent for two VASPs only. This enables us to identify gaps in the data collection and propose strategies to address them. We remark that any entity in charge of auditing requires proof that a VASP actually controls the funds associated with its on-chain wallets. It is also important to report fiat and cryptoasset and liability positions broken down by asset types at a reasonable frequency.

{{</citation>}}


## physics.comp-ph (1)



### (140/151) Physics-Preserving AI-Accelerated Simulations of Plasma Turbulence (Robin Greif et al., 2023)

{{<citation>}}

Robin Greif, Frank Jenko, Nils Thuerey. (2023)  
**Physics-Preserving AI-Accelerated Simulations of Plasma Turbulence**  

---
Primary Category: physics.comp-ph  
Categories: cs-AI, physics-comp-ph, physics-plasm-ph, physics.comp-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16400v1)  

---


**ABSTRACT**  
Turbulence in fluids, gases, and plasmas remains an open problem of both practical and fundamental importance. Its irreducible complexity usually cannot be tackled computationally in a brute-force style. Here, we combine Large Eddy Simulation (LES) techniques with Machine Learning (ML) to retain only the largest dynamics explicitly, while small-scale dynamics are described by an ML-based sub-grid-scale model. Applying this novel approach to self-driven plasma turbulence allows us to remove large parts of the inertial range, reducing the computational effort by about three orders of magnitude, while retaining the statistical physical properties of the turbulent system.

{{</citation>}}


## astro-ph.SR (1)



### (141/151) Astroconformer: The Prospects of Analyzing Stellar Light Curves with Transformer-Based Deep Learning Models (Jia-Shu Pan et al., 2023)

{{<citation>}}

Jia-Shu Pan, Yuan-Sen Ting, Jie Yu. (2023)  
**Astroconformer: The Prospects of Analyzing Stellar Light Curves with Transformer-Based Deep Learning Models**  

---
Primary Category: astro-ph.SR  
Categories: astro-ph-EP, astro-ph-IM, astro-ph-SR, astro-ph.SR, cs-LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.16316v1)  

---


**ABSTRACT**  
Light curves of stars encapsulate a wealth of information about stellar oscillations and granulation, thereby offering key insights into the internal structure and evolutionary state of stars. Conventional asteroseismic techniques have been largely confined to power spectral analysis, neglecting the valuable phase information contained within light curves. While recent machine learning applications in asteroseismology utilizing Convolutional Neural Networks (CNNs) have successfully inferred stellar attributes from light curves, they are often limited by the local feature extraction inherent in convolutional operations. To circumvent these constraints, we present $\textit{Astroconformer}$, a Transformer-based deep learning framework designed to capture long-range dependencies in stellar light curves. Our empirical analysis, which focuses on estimating surface gravity ($\log g$), is grounded in a carefully curated dataset derived from $\textit{Kepler}$ light curves. These light curves feature asteroseismic $\log g$ values spanning from 0.2 to 4.4. Our results underscore that, in the regime where the training data is abundant, $\textit{Astroconformer}$ attains a root-mean-square-error (RMSE) of 0.017 dex around $\log g \approx 3 $. Even in regions where training data are sparse, the RMSE can reach 0.1 dex. It outperforms not only the K-nearest neighbor-based model ($\textit{The SWAN}$) but also state-of-the-art CNNs. Ablation studies confirm that the efficacy of the models in this particular task is strongly influenced by the size of their receptive fields, with larger receptive fields correlating with enhanced performance. Moreover, we find that the attention mechanisms within $\textit{Astroconformer}$ are well-aligned with the inherent characteristics of stellar oscillations and granulation present in the light curves.

{{</citation>}}


## q-bio.BM (1)



### (142/151) 3D-Mol: A Novel Contrastive Learning Framework for Molecular Property Prediction with 3D Information (Taojie Kuang et al., 2023)

{{<citation>}}

Taojie Kuang, Yiming Ren, Zhixiang Ren. (2023)  
**3D-Mol: A Novel Contrastive Learning Framework for Molecular Property Prediction with 3D Information**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.17366v1)  

---


**ABSTRACT**  
Molecular property prediction offers an effective and efficient approach for early screening and optimization of drug candidates. Although deep learning based methods have made notable progress, most existing works still do not fully utilize 3D spatial information. This can lead to a single molecular representation representing multiple actual molecules. To address these issues, we propose a novel 3D structure-based molecular modeling method named 3D-Mol. In order to accurately represent complete spatial structure, we design a novel encoder to extract 3D features by deconstructing the molecules into three geometric graphs. In addition, we use 20M unlabeled data to pretrain our model by contrastive learning. We consider conformations with the same topological structure as positive pairs and the opposites as negative pairs, while the weight is determined by the dissimilarity between the conformations. We compare 3D-Mol with various state-of-the-art (SOTA) baselines on 7 benchmarks and demonstrate our outstanding performance in 5 benchmarks.

{{</citation>}}


## cs.CE (1)



### (143/151) TaxAI: A Dynamic Economic Simulator and Benchmark for Multi-Agent Reinforcement Learning (Qirui Mi et al., 2023)

{{<citation>}}

Qirui Mi, Siyu Xia, Yan Song, Haifeng Zhang, Shenghao Zhu, Jun Wang. (2023)  
**TaxAI: A Dynamic Economic Simulator and Benchmark for Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16307v1)  

---


**ABSTRACT**  
Taxation and government spending are crucial tools for governments to promote economic growth and maintain social equity. However, the difficulty in accurately predicting the dynamic strategies of diverse self-interested households presents a challenge for governments to implement effective tax policies. Given its proficiency in modeling other agents in partially observable environments and adaptively learning to find optimal policies, Multi-Agent Reinforcement Learning (MARL) is highly suitable for solving dynamic games between the government and numerous households. Although MARL shows more potential than traditional methods such as the genetic algorithm and dynamic programming, there is a lack of large-scale multi-agent reinforcement learning economic simulators. Therefore, we propose a MARL environment, named \textbf{TaxAI}, for dynamic games involving $N$ households, government, firms, and financial intermediaries based on the Bewley-Aiyagari economic model. Our study benchmarks 2 traditional economic methods with 7 MARL methods on TaxAI, demonstrating the effectiveness and superiority of MARL algorithms. Moreover, TaxAI's scalability in simulating dynamic interactions between the government and 10,000 households, coupled with real-data calibration, grants it a substantial improvement in scale and reality over existing simulators. Therefore, TaxAI is the most realistic economic simulator, which aims to generate feasible recommendations for governments and individuals.

{{</citation>}}


## cs.DB (2)



### (144/151) A Framework to Assess Knowledge Graphs Accountability (Jennie Andersen et al., 2023)

{{<citation>}}

Jennie Andersen, Sylvie Cazalens, Philippe Lamarre, Pierre Maillot. (2023)  
**A Framework to Assess Knowledge Graphs Accountability**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.16285v1)  

---


**ABSTRACT**  
Knowledge Graphs (KGs), and Linked Open Data in particular, enable the generation and exchange of more and more information on the Web. In order to use and reuse these data properly, the presence of accountability information is essential. Accountability requires specific and accurate information about people's responsibilities and actions. In this article, we define KGAcc, a framework dedicated to the assessment of RDF graphs accountability. It consists of accountability requirements and a measure of accountability for KGs. Then, we evaluate KGs from the LOD cloud and describe the results obtained. Finally, we compare our approach with data quality and FAIR assessment frameworks to highlight the differences.

{{</citation>}}


### (145/151) Sampling Methods for Inner Product Sketching (Majid Daliri et al., 2023)

{{<citation>}}

Majid Daliri, Juliana Freire, Christopher Musco, Aécio Santos, Haoxiang Zhang. (2023)  
**Sampling Methods for Inner Product Sketching**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs-DS, cs.DB  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.16157v2)  

---


**ABSTRACT**  
Recently, Bessa et al. (PODS 2023) showed that sketches based on coordinated weighted sampling theoretically and empirically outperform popular linear sketching methods like Johnson-Lindentrauss projection and CountSketch for the ubiquitous problem of inner product estimation. Despite decades of literature on such sampling methods, this observation seems to have been overlooked. We further develop the finding by presenting and analyzing two alternative sampling-based inner product sketching methods. In contrast to the computationally expensive algorithm in Bessa et al., our methods run in linear time (to compute the sketch) and perform better in practice, significantly beating linear sketching on a variety of tasks. For example, they provide state-of-the-art results for estimating the correlation between columns in unjoined tables, a problem that we show how to reduce to inner product estimation in a black-box way. While based on known sampling techniques (threshold and priority sampling) we introduce significant new theoretical analysis to prove approximation guarantees for our methods.

{{</citation>}}


## cs.GT (1)



### (146/151) Cooperation Dynamics in Multi-Agent Systems: Exploring Game-Theoretic Scenarios with Mean-Field Equilibria (Vaigarai Sathi et al., 2023)

{{<citation>}}

Vaigarai Sathi, Sabahat Shaik, Jaswanth Nidamanuri. (2023)  
**Cooperation Dynamics in Multi-Agent Systems: Exploring Game-Theoretic Scenarios with Mean-Field Equilibria**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-GT, cs.GT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16263v2)  

---


**ABSTRACT**  
Cooperation is fundamental in Multi-Agent Systems (MAS) and Multi-Agent Reinforcement Learning (MARL), often requiring agents to balance individual gains with collective rewards. In this regard, this paper aims to investigate strategies to invoke cooperation in game-theoretic scenarios, namely the Iterated Prisoner's Dilemma, where agents must optimize both individual and group outcomes. Existing cooperative strategies are analyzed for their effectiveness in promoting group-oriented behavior in repeated games. Modifications are proposed where encouraging group rewards will also result in a higher individual gain, addressing real-world dilemmas seen in distributed systems. The study extends to scenarios with exponentially growing agent populations ($N \longrightarrow +\infty$), where traditional computation and equilibrium determination are challenging. Leveraging mean-field game theory, equilibrium solutions and reward structures are established for infinitely large agent sets in repeated games. Finally, practical insights are offered through simulations using the Multi Agent-Posthumous Credit Assignment trainer, and the paper explores adapting simulation algorithms to create scenarios favoring cooperation for group rewards. These practical implementations bridge theoretical concepts with real-world applications.

{{</citation>}}


## physics.chem-ph (1)



### (147/151) Language models in molecular discovery (Nikita Janakarajan et al., 2023)

{{<citation>}}

Nikita Janakarajan, Tim Erdmann, Sarath Swaminathan, Teodoro Laino, Jannis Born. (2023)  
**Language models in molecular discovery**  

---
Primary Category: physics.chem-ph  
Categories: cs-AI, cs-CL, cs-LG, physics-chem-ph, physics.chem-ph, q-bio-BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16235v1)  

---


**ABSTRACT**  
The success of language models, especially transformer-based architectures, has trickled into other domains giving rise to "scientific language models" that operate on small molecules, proteins or polymers. In chemistry, language models contribute to accelerating the molecule discovery cycle as evidenced by promising recent findings in early-stage drug discovery. Here, we review the role of language models in molecular discovery, underlining their strength in de novo drug design, property prediction and reaction chemistry. We highlight valuable open-source software assets thus lowering the entry barrier to the field of scientific language modeling. Last, we sketch a vision for future molecular design that combines a chatbot interface with access to computational chemistry tools. Our contribution serves as a valuable resource for researchers, chemists, and AI enthusiasts interested in understanding how language models can and will be used to accelerate chemical discovery.

{{</citation>}}


## eess.SY (1)



### (148/151) Adaptive Real-Time Numerical Differentiation with Variable-Rate Forgetting and Exponential Resetting (Shashank Verma et al., 2023)

{{<citation>}}

Shashank Verma, Brian Lai, Dennis S. Bernstein. (2023)  
**Adaptive Real-Time Numerical Differentiation with Variable-Rate Forgetting and Exponential Resetting**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SP, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16159v1)  

---


**ABSTRACT**  
Digital PID control requires a differencing operation to implement the D gain. In order to suppress the effects of noisy data, the traditional approach is to filter the data, where the frequency response of the filter is adjusted manually based on the characteristics of the sensor noise. The present paper considers the case where the characteristics of the sensor noise change over time in an unknown way. This problem is addressed by applying adaptive real-time numerical differentiation based on adaptive input and state estimation (AISE). The contribution of this paper is to extend AISE to include variable-rate forgetting with exponential resetting, which allows AISE to more rapidly respond to changing noise characteristics while enforcing the boundedness of the covariance matrix used in recursive least squares.

{{</citation>}}


## cs.SE (2)



### (149/151) Let's Chat to Find the APIs: Connecting Human, LLM and Knowledge Graph through AI Chain (Qing Huang et al., 2023)

{{<citation>}}

Qing Huang, Zhenyu Wan, Zhenchang Xing, Changjing Wang, Jieshan Chen, Xiwei Xu, Qinghua Lu. (2023)  
**Let's Chat to Find the APIs: Connecting Human, LLM and Knowledge Graph through AI Chain**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.16134v1)  

---


**ABSTRACT**  
API recommendation methods have evolved from literal and semantic keyword matching to query expansion and query clarification. The latest query clarification method is knowledge graph (KG)-based, but limitations include out-of-vocabulary (OOV) failures and rigid question templates. To address these limitations, we propose a novel knowledge-guided query clarification approach for API recommendation that leverages a large language model (LLM) guided by KG. We utilize the LLM as a neural knowledge base to overcome OOV failures, generating fluent and appropriate clarification questions and options. We also leverage the structured API knowledge and entity relationships stored in the KG to filter out noise, and transfer the optimal clarification path from KG to the LLM, increasing the efficiency of the clarification process. Our approach is designed as an AI chain that consists of five steps, each handled by a separate LLM call, to improve accuracy, efficiency, and fluency for query clarification in API recommendation. We verify the usefulness of each unit in our AI chain, which all received high scores close to a perfect 5. When compared to the baselines, our approach shows a significant improvement in MRR, with a maximum increase of 63.9% higher when the query statement is covered in KG and 37.2% when it is not. Ablation experiments reveal that the guidance of knowledge in the KG and the knowledge-guided pathfinding strategy are crucial for our approach's performance, resulting in a 19.0% and 22.2% increase in MAP, respectively. Our approach demonstrates a way to bridge the gap between KG and LLM, effectively compensating for the strengths and weaknesses of both.

{{</citation>}}


### (150/151) Test-Case-Driven Programming Understanding in Large Language Models for Better Code Generation (Zhao Tian et al., 2023)

{{<citation>}}

Zhao Tian, Junjie Chen. (2023)  
**Test-Case-Driven Programming Understanding in Large Language Models for Better Code Generation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.16120v1)  

---


**ABSTRACT**  
Code generation is to automatically generate source code conforming to a given programming specification, which has received extensive attention especially with the development of large language models (LLMs). Due to the inherent difficulty of code generation, the code generated by LLMs may be also not aligned with the specification. To improve the perfor mance of LLMs in code generation, some Chain of Thought (CoT) techniques have been proposed to guide LLMs for programming understanding before code generation. However, they are still hard to figure out complicated programming logic according to the (concise) specification, leadingto unsatisfactory code generation performance. In this work, we propose the first test-case-driven CoT technique, called TCoT, to further enhance the ability of LLMs in code generation. It understands the programming specification from the novel perspective of test cases, which is aligned with human practice by using examples to understand complicated problems. Due to the existence of the expected output specified in a test case, TCoT can instantly check the correctness of the programming understanding and then refine it to be as correct as possible before code generation. In this way, it is more likely to generate correct code. Our evaluation on 6 datasets and 14 baselines demonstrates the effectiveness of TCoT. For example, TCoT improves ChatGPT by 13.93%~69.44% in terms of Pass@1 (measuring the ratio of programming problems for which the generated code passes all test cases), and outperforms the existing CoT technique with the improvement of 12.14%~53.72% in terms of Pass@1.

{{</citation>}}


## eess.AS (1)



### (151/151) Hierarchical Cross-Modality Knowledge Transfer with Sinkhorn Attention for CTC-based ASR (Xugang Lu et al., 2023)

{{<citation>}}

Xugang Lu, Peng Shen, Yu Tsao, Hisashi Kawai. (2023)  
**Hierarchical Cross-Modality Knowledge Transfer with Sinkhorn Attention for CTC-based ASR**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2309.16093v1)  

---


**ABSTRACT**  
Due to the modality discrepancy between textual and acoustic modeling, efficiently transferring linguistic knowledge from a pretrained language model (PLM) to acoustic encoding for automatic speech recognition (ASR) still remains a challenging task. In this study, we propose a cross-modality knowledge transfer (CMKT) learning framework in a temporal connectionist temporal classification (CTC) based ASR system where hierarchical acoustic alignments with the linguistic representation are applied. Additionally, we propose the use of Sinkhorn attention in cross-modality alignment process, where the transformer attention is a special case of this Sinkhorn attention process. The CMKT learning is supposed to compel the acoustic encoder to encode rich linguistic knowledge for ASR. On the AISHELL-1 dataset, with CTC greedy decoding for inference (without using any language model), we achieved state-of-the-art performance with 3.64% and 3.94% character error rates (CERs) for the development and test sets, which corresponding to relative improvements of 34.18% and 34.88% compared to the baseline CTC-ASR system, respectively.

{{</citation>}}
