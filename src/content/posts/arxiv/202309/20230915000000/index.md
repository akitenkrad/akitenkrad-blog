---
draft: false
title: "arXiv @ 2023.09.15"
date: 2023-09-15
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.15"
    identifier: arxiv_20230915
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (8)](#csro-8)
- [quant-ph (2)](#quant-ph-2)
- [cs.CV (18)](#cscv-18)
- [cs.NI (2)](#csni-2)
- [cs.CL (22)](#cscl-22)
- [cs.HC (5)](#cshc-5)
- [cs.LG (17)](#cslg-17)
- [cs.SD (3)](#cssd-3)
- [stat.ML (1)](#statml-1)
- [eess.AS (1)](#eessas-1)
- [cs.CR (7)](#cscr-7)
- [cs.SE (1)](#csse-1)
- [cs.MA (1)](#csma-1)
- [cs.AI (3)](#csai-3)
- [cs.IR (1)](#csir-1)
- [physics.data-an (1)](#physicsdata-an-1)
- [eess.SY (1)](#eesssy-1)

## cs.RO (8)



### (1/94) Curriculum-based Sensing Reduction in Simulation to Real-World Transfer for In-hand Manipulation (Lingfeng Tao et al., 2023)

{{<citation>}}

Lingfeng Tao, Jiucai Zhang, Qiaojie Zheng, Xiaoli Zhang. (2023)  
**Curriculum-based Sensing Reduction in Simulation to Real-World Transfer for In-hand Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.07350v1)  

---


**ABSTRACT**  
Simulation to Real-World Transfer allows affordable and fast training of learning-based robots for manipulation tasks using Deep Reinforcement Learning methods. Currently, Sim2Real uses Asymmetric Actor-Critic approaches to reduce the rich idealized features in simulation to the accessible ones in the real world. However, the feature reduction from the simulation to the real world is conducted through an empirically defined one-step curtail. Small feature reduction does not sufficiently remove the actor's features, which may still cause difficulty setting up the physical system, while large feature reduction may cause difficulty and inefficiency in training. To address this issue, we proposed Curriculum-based Sensing Reduction to enable the actor to start with the same rich feature space as the critic and then get rid of the hard-to-extract features step-by-step for higher training performance and better adaptation for real-world feature space. The reduced features are replaced with random signals from a Deep Random Generator to remove the dependency between the output and the removed features and avoid creating new dependencies. The methods are evaluated on the Allegro robot hand in a real-world in-hand manipulation task. The results show that our methods have faster training and higher task performance than baselines and can solve real-world tasks when selected tactile features are reduced.

{{</citation>}}


### (2/94) Stable In-hand Manipulation with Finger Specific Multi-agent Shadow Reward (Lingfeng Tao et al., 2023)

{{<citation>}}

Lingfeng Tao, Jiucai Zhang, Xiaoli Zhang. (2023)  
**Stable In-hand Manipulation with Finger Specific Multi-agent Shadow Reward**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.07349v1)  

---


**ABSTRACT**  
Deep Reinforcement Learning has shown its capability to solve the high degrees of freedom in control and the complex interaction with the object in the multi-finger dexterous in-hand manipulation tasks. Current DRL approaches prefer sparse rewards to dense rewards for the ease of training but lack behavior constraints during the manipulation process, leading to aggressive and unstable policies that are insufficient for safety-critical in-hand manipulation tasks. Dense rewards can regulate the policy to learn stable manipulation behaviors with continuous reward constraints but are hard to empirically define and slow to converge optimally. This work proposes the Finger-specific Multi-agent Shadow Reward (FMSR) method to determine the stable manipulation constraints in the form of dense reward based on the state-action occupancy measure, a general utility of DRL that is approximated during the learning process. Information Sharing (IS) across neighboring agents enables consensus training to accelerate the convergence. The methods are evaluated in two in-hand manipulation tasks on the Shadow Hand. The results show FMSR+IS converges faster in training, achieving a higher task success rate and better manipulation stability than conventional dense reward. The comparison indicates FMSR+IS achieves a comparable success rate even with the behavior constraint but much better manipulation stability than the policy trained with a sparse reward.

{{</citation>}}


### (3/94) Efficient Reinforcement Learning for Jumping Monopods (Riccardo Bussola et al., 2023)

{{<citation>}}

Riccardo Bussola, Michele Focchi, Andrea Del Prete, Daniele Fontanelli, Luigi Palopoli. (2023)  
**Efficient Reinforcement Learning for Jumping Monopods**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.07038v1)  

---


**ABSTRACT**  
In this work, we consider the complex control problem of making a monopod reach a target with a jump. The monopod can jump in any direction and the terrain underneath its foot can be uneven. This is a template of a much larger class of problems, which are extremely challenging and computationally expensive to solve using standard optimisation-based techniques. Reinforcement Learning (RL) could be an interesting alternative, but the application of an end-to-end approach in which the controller must learn everything from scratch, is impractical. The solution advocated in this paper is to guide the learning process within an RL framework by injecting physical knowledge. This expedient brings to widespread benefits, such as a drastic reduction of the learning time, and the ability to learn and compensate for possible errors in the low-level controller executing the motion. We demonstrate the advantage of our approach with respect to both optimization-based and end-to-end RL approaches.

{{</citation>}}


### (4/94) Learning to Explore Indoor Environments using Autonomous Micro Aerial Vehicles (Yuezhan Tao et al., 2023)

{{<citation>}}

Yuezhan Tao, Eran Iceland, Beiming Li, Elchanan Zwecher, Uri Heinemann, Avraham Cohen, Amir Avni, Oren Gal, Ariel Barel, Vijay Kumar. (2023)  
**Learning to Explore Indoor Environments using Autonomous Micro Aerial Vehicles**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06986v1)  

---


**ABSTRACT**  
In this paper, we address the challenge of exploring unknown indoor aerial environments using autonomous aerial robots with Size Weight and Power (SWaP) constraints. The SWaP constraints induce limits on mission time requiring efficiency in exploration. We present a novel exploration framework that uses Deep Learning (DL) to predict the most likely indoor map given the previous observations, and Deep Reinforcement Learning (DRL) for exploration, designed to run on modern SWaP constraints neural processors. The DL-based map predictor provides a prediction of the occupancy of the unseen environment while the DRL-based planner determines the best navigation goals that can be safely reached to provide the most information. The two modules are tightly coupled and run onboard allowing the vehicle to safely map an unknown environment. Extensive experimental and simulation results show that our approach surpasses state-of-the-art methods by 50-60% in efficiency, which we measure by the fraction of the explored space as a function of the length of the trajectory traveled.

{{</citation>}}


### (5/94) Utilizing Hybrid Trajectory Prediction Models to Recognize Highly Interactive Traffic Scenarios (Maximilian Zipfl et al., 2023)

{{<citation>}}

Maximilian Zipfl, Sven Spickermann, J. Marius Zöllner. (2023)  
**Utilizing Hybrid Trajectory Prediction Models to Recognize Highly Interactive Traffic Scenarios**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.06887v1)  

---


**ABSTRACT**  
Autonomous vehicles hold great promise in improving the future of transportation. The driving models used in these vehicles are based on neural networks, which can be difficult to validate. However, ensuring the safety of these models is crucial. Traditional field tests can be costly, time-consuming, and dangerous. To address these issues, scenario-based closed-loop simulations can simulate many hours of vehicle operation in a shorter amount of time and allow for specific investigation of important situations. Nonetheless, the detection of relevant traffic scenarios that also offer substantial testing benefits remains a significant challenge. To address this need, in this paper we build an imitation learning based trajectory prediction for traffic participants. We combine an image-based (CNN) approach to represent spatial environmental factors and a graph-based (GNN) approach to specifically represent relations between traffic participants. In our understanding, traffic scenes that are highly interactive due to the network's significant utilization of the social component are more pertinent for a validation process. Therefore, we propose to use the activity of such sub networks as a measure of interactivity of a traffic scene. We evaluate our model using a motion dataset and discuss the value of the relationship information with respect to different traffic situations.

{{</citation>}}


### (6/94) Lavender Autonomous Navigation with Semantic Segmentation at the Edge (Alessandro Navone et al., 2023)

{{<citation>}}

Alessandro Navone, Fabrizio Romanelli, Marco Ambrosio, Mauro Martini, Simone Angarano, Marcello Chiaberge. (2023)  
**Lavender Autonomous Navigation with Semantic Segmentation at the Edge**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.06863v1)  

---


**ABSTRACT**  
Achieving success in agricultural activities heavily relies on precise navigation in row crop fields. Recently, segmentation-based navigation has emerged as a reliable technique when GPS-based localization is unavailable or higher accuracy is needed due to vegetation or unfavorable weather conditions. It also comes in handy when plants are growing rapidly and require an online adaptation of the navigation algorithm. This work applies a segmentation-based visual agnostic navigation algorithm to lavender fields, considering both simulation and real-world scenarios. The effectiveness of this approach is validated through a wide set of experimental tests, which show the capability of the proposed solution to generalize over different scenarios and provide highly-reliable results.

{{</citation>}}


### (7/94) Time-Optimal Gate-Traversing Planner for Autonomous Drone Racing (Chao Qin et al., 2023)

{{<citation>}}

Chao Qin, Maxime S. J. Michet, Jingxiang Chen, Hugh H. -T. Liu. (2023)  
**Time-Optimal Gate-Traversing Planner for Autonomous Drone Racing**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.06837v1)  

---


**ABSTRACT**  
Time-minimum trajectories through race tracks are determined by the drone's capability as well as the configuration of all gates (e.g., their shapes, sizes, and orientations). However, prior works neglect the impact of the gate configuration and formulate drone racing as a waypoint flight task, leading to conservative waypoint selection through each gate. We present a novel time-optimal planner that can account for gate constraints explicitly, enabling quadrotors to follow the most time-efficient waypoints at their single-rotor-thrust limits in tracks with hybrid gate types. Our approach provides comparable solution quality to the state-of-the-art but with a computation time orders of magnitude faster. Furthermore, the proposed framework allows users to customize gate constraints such as tunnels by concatenating existing gate classes, enabling high-fidelity race track modeling. Owing to the superior computation efficiency and flexibility, we can generate optimal racing trajectories for complex race tracks with tens or even hundreds of gates with distinct shapes. We validate our method in real-world flights and demonstrate that faster lap times can be produced by using gate constraints instead of waypoint constraints.

{{</citation>}}


### (8/94) Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics (Jiayang Song et al., 2023)

{{<citation>}}

Jiayang Song, Zhehua Zhou, Jiawei Liu, Chunrong Fang, Zhan Shu, Lei Ma. (2023)  
**Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06687v1)  

---


**ABSTRACT**  
Although Deep Reinforcement Learning (DRL) has achieved notable success in numerous robotic applications, designing a high-performing reward function remains a challenging task that often requires substantial manual input. Recently, Large Language Models (LLMs) have been extensively adopted to address tasks demanding in-depth common-sense knowledge, such as reasoning and planning. Recognizing that reward function design is also inherently linked to such knowledge, LLM offers a promising potential in this context. Motivated by this, we propose in this work a novel LLM framework with a self-refinement mechanism for automated reward function design. The framework commences with the LLM formulating an initial reward function based on natural language inputs. Then, the performance of the reward function is assessed, and the results are presented back to the LLM for guiding its self-refinement process. We examine the performance of our proposed framework through a variety of continuous robotic control tasks across three diverse robotic systems. The results indicate that our LLM-designed reward functions are able to rival or even surpass manually designed reward functions, highlighting the efficacy and applicability of our approach.

{{</citation>}}


## quant-ph (2)



### (9/94) Efficient quantum recurrent reinforcement learning via quantum reservoir computing (Samuel Yen-Chi Chen, 2023)

{{<citation>}}

Samuel Yen-Chi Chen. (2023)  
**Efficient quantum recurrent reinforcement learning via quantum reservoir computing**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-ET, cs-LG, cs-NE, quant-ph, quant-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.07339v1)  

---


**ABSTRACT**  
Quantum reinforcement learning (QRL) has emerged as a framework to solve sequential decision-making tasks, showcasing empirical quantum advantages. A notable development is through quantum recurrent neural networks (QRNNs) for memory-intensive tasks such as partially observable environments. However, QRL models incorporating QRNN encounter challenges such as inefficient training of QRL with QRNN, given that the computation of gradients in QRNN is both computationally expensive and time-consuming. This work presents a novel approach to address this challenge by constructing QRL agents utilizing QRNN-based reservoirs, specifically employing quantum long short-term memory (QLSTM). QLSTM parameters are randomly initialized and fixed without training. The model is trained using the asynchronous advantage actor-aritic (A3C) algorithm. Through numerical simulations, we validate the efficacy of our QLSTM-Reservoir RL framework. Its performance is assessed on standard benchmarks, demonstrating comparable results to a fully trained QLSTM RL model with identical architecture and training settings.

{{</citation>}}


### (10/94) Deep Quantum Graph Dreaming: Deciphering Neural Network Insights into Quantum Experiments (Tareq Jaouni et al., 2023)

{{<citation>}}

Tareq Jaouni, Sören Arlt, Carlos Ruiz-Gonzalez, Ebrahim Karimi, Xuemei Gu, Mario Krenn. (2023)  
**Deep Quantum Graph Dreaming: Deciphering Neural Network Insights into Quantum Experiments**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-LG, quant-ph, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.07056v1)  

---


**ABSTRACT**  
Despite their promise to facilitate new scientific discoveries, the opaqueness of neural networks presents a challenge in interpreting the logic behind their findings. Here, we use a eXplainable-AI (XAI) technique called $inception$ or $deep$ $dreaming$, which has been invented in machine learning for computer vision. We use this techniques to explore what neural networks learn about quantum optics experiments. Our story begins by training a deep neural networks on the properties of quantum systems. Once trained, we "invert" the neural network -- effectively asking how it imagines a quantum system with a specific property, and how it would continuously modify the quantum system to change a property. We find that the network can shift the initial distribution of properties of the quantum system, and we can conceptualize the learned strategies of the neural network. Interestingly, we find that, in the first layers, the neural network identifies simple properties, while in the deeper ones, it can identify complex quantum structures and even quantum entanglement. This is in reminiscence of long-understood properties known in computer vision, which we now identify in a complex natural science task. Our approach could be useful in a more interpretable way to develop new advanced AI-based scientific discovery techniques in quantum physics.

{{</citation>}}


## cs.CV (18)



### (11/94) Automated Assessment of Critical View of Safety in Laparoscopic Cholecystectomy (Yunfan Li et al., 2023)

{{<citation>}}

Yunfan Li, Himanshu Gupta, Haibin Ling, IV Ramakrishnan, Prateek Prasanna, Georgios Georgakis, Aaron Sasson. (2023)  
**Automated Assessment of Critical View of Safety in Laparoscopic Cholecystectomy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.07330v1)  

---


**ABSTRACT**  
Cholecystectomy (gallbladder removal) is one of the most common procedures in the US, with more than 1.2M procedures annually. Compared with classical open cholecystectomy, laparoscopic cholecystectomy (LC) is associated with significantly shorter recovery period, and hence is the preferred method. However, LC is also associated with an increase in bile duct injuries (BDIs), resulting in significant morbidity and mortality. The primary cause of BDIs from LCs is misidentification of the cystic duct with the bile duct. Critical view of safety (CVS) is the most effective of safety protocols, which is said to be achieved during the surgery if certain criteria are met. However, due to suboptimal understanding and implementation of CVS, the BDI rates have remained stable over the last three decades. In this paper, we develop deep-learning techniques to automate the assessment of CVS in LCs. An innovative aspect of our research is on developing specialized learning techniques by incorporating domain knowledge to compensate for the limited training data available in practice. In particular, our CVS assessment process involves a fusion of two segmentation maps followed by an estimation of a certain region of interest based on anatomical structures close to the gallbladder, and then finally determination of each of the three CVS criteria via rule-based assessment of structural information. We achieved a gain of over 11.8% in mIoU on relevant classes with our two-stream semantic segmentation approach when compared to a single-model baseline, and 1.84% in mIoU with our proposed Sobel loss function when compared to a Transformer-based baseline model. For CVS criteria, we achieved up to 16% improvement and, for the overall CVS assessment, we achieved 5% improvement in balanced accuracy compared to DeepCVS under the same experiment settings.

{{</citation>}}


### (12/94) SupFusion: Supervised LiDAR-Camera Fusion for 3D Object Detection (Yiran Qin et al., 2023)

{{<citation>}}

Yiran Qin, Chaoqun Wang, Zijian Kang, Ningning Ma, Zhen Li, Ruimao Zhang. (2023)  
**SupFusion: Supervised LiDAR-Camera Fusion for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.07084v1)  

---


**ABSTRACT**  
In this paper, we propose a novel training strategy called SupFusion, which provides an auxiliary feature level supervision for effective LiDAR-Camera fusion and significantly boosts detection performance. Our strategy involves a data enhancement method named Polar Sampling, which densifies sparse objects and trains an assistant model to generate high-quality features as the supervision. These features are then used to train the LiDAR-Camera fusion model, where the fusion feature is optimized to simulate the generated high-quality features. Furthermore, we propose a simple yet effective deep fusion module, which contiguously gains superior performance compared with previous fusion methods with SupFusion strategy. In such a manner, our proposal shares the following advantages. Firstly, SupFusion introduces auxiliary feature-level supervision which could boost LiDAR-Camera detection performance without introducing extra inference costs. Secondly, the proposed deep fusion could continuously improve the detector's abilities. Our proposed SupFusion and deep fusion module is plug-and-play, we make extensive experiments to demonstrate its effectiveness. Specifically, we gain around 2% 3D mAP improvements on KITTI benchmark based on multiple LiDAR-Camera 3D detectors.

{{</citation>}}


### (13/94) FAIR: Frequency-aware Image Restoration for Industrial Visual Anomaly Detection (Tongkun Liu et al., 2023)

{{<citation>}}

Tongkun Liu, Bing Li, Xiao Du, Bingke Jiang, Leqi Geng, Feiyang Wang, Zhuo Zhao. (2023)  
**FAIR: Frequency-aware Image Restoration for Industrial Visual Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.07068v1)  

---


**ABSTRACT**  
Image reconstruction-based anomaly detection models are widely explored in industrial visual inspection. However, existing models usually suffer from the trade-off between normal reconstruction fidelity and abnormal reconstruction distinguishability, which damages the performance. In this paper, we find that the above trade-off can be better mitigated by leveraging the distinct frequency biases between normal and abnormal reconstruction errors. To this end, we propose Frequency-aware Image Restoration (FAIR), a novel self-supervised image restoration task that restores images from their high-frequency components. It enables precise reconstruction of normal patterns while mitigating unfavorable generalization to anomalies. Using only a simple vanilla UNet, FAIR achieves state-of-the-art performance with higher efficiency on various defect detection datasets. Code: https://github.com/liutongkun/FAIR.

{{</citation>}}


### (14/94) Aggregating Long-term Sharp Features via Hybrid Transformers for Video Deblurring (Dongwei Ren et al., 2023)

{{<citation>}}

Dongwei Ren, Wei Shang, Yi Yang, Wangmeng Zuo. (2023)  
**Aggregating Long-term Sharp Features via Hybrid Transformers for Video Deblurring**  

---
Primary Category: cs.CV  
Categories: I-4-3, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.07054v1)  

---


**ABSTRACT**  
Video deblurring methods, aiming at recovering consecutive sharp frames from a given blurry video, usually assume that the input video suffers from consecutively blurry frames. However, in real-world blurry videos taken by modern imaging devices, sharp frames usually appear in the given video, thus making temporal long-term sharp features available for facilitating the restoration of a blurry frame. In this work, we propose a video deblurring method that leverages both neighboring frames and present sharp frames using hybrid Transformers for feature aggregation. Specifically, we first train a blur-aware detector to distinguish between sharp and blurry frames. Then, a window-based local Transformer is employed for exploiting features from neighboring frames, where cross attention is beneficial for aggregating features from neighboring frames without explicit spatial alignment. To aggregate long-term sharp features from detected sharp frames, we utilize a global Transformer with multi-scale matching capability. Moreover, our method can easily be extended to event-driven video deblurring by incorporating an event fusion module into the global Transformer. Extensive experiments on benchmark datasets demonstrate that our proposed method outperforms state-of-the-art video deblurring methods as well as event-driven video deblurring methods in terms of quantitative metrics and visual quality. The source code and trained models are available at https://github.com/shangwei5/STGTN.

{{</citation>}}


### (15/94) Instance Adaptive Prototypical Contrastive Embedding for Generalized Zero Shot Learning (Riti Paul et al., 2023)

{{<citation>}}

Riti Paul, Sahil Vora, Baoxin Li. (2023)  
**Instance Adaptive Prototypical Contrastive Embedding for Generalized Zero Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.06987v2)  

---


**ABSTRACT**  
Generalized zero-shot learning(GZSL) aims to classify samples from seen and unseen labels, assuming unseen labels are not accessible during training. Recent advancements in GZSL have been expedited by incorporating contrastive-learning-based (instance-based) embedding in generative networks and leveraging the semantic relationship between data points. However, existing embedding architectures suffer from two limitations: (1) limited discriminability of synthetic features' embedding without considering fine-grained cluster structures; (2) inflexible optimization due to restricted scaling mechanisms on existing contrastive embedding networks, leading to overlapped representations in the embedding space. To enhance the quality of representations in the embedding space, as mentioned in (1), we propose a margin-based prototypical contrastive learning embedding network that reaps the benefits of prototype-data (cluster quality enhancement) and implicit data-data (fine-grained representations) interaction while providing substantial cluster supervision to the embedding network and the generator. To tackle (2), we propose an instance adaptive contrastive loss that leads to generalized representations for unseen labels with increased inter-class margin. Through comprehensive experimental evaluation, we show that our method can outperform the current state-of-the-art on three benchmark datasets. Our approach also consistently achieves the best unseen performance in the GZSL setting.

{{</citation>}}


### (16/94) DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision (Xiangchen Yin et al., 2023)

{{<citation>}}

Xiangchen Yin, Zhenda Yu, Xin Gao, Ran Ju, Xiao Sun, Xinyu Zhang. (2023)  
**DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.06941v1)  

---


**ABSTRACT**  
The goal of low-light image enhancement is to restore the color and details of the image and is of great significance for high-level visual tasks in autonomous driving. However, it is difficult to restore the lost details in the dark area by relying only on the RGB domain. In this paper we introduce frequency as a new clue into the network and propose a novel DCT-driven enhancement transformer (DEFormer). First, we propose a learnable frequency branch (LFB) for frequency enhancement contains DCT processing and curvature-based frequency enhancement (CFE). CFE calculates the curvature of each channel to represent the detail richness of different frequency bands, then we divides the frequency features, which focuses on frequency bands with richer textures. In addition, we propose a cross domain fusion (CDF) for reducing the differences between the RGB domain and the frequency domain. We also adopt DEFormer as a preprocessing in dark detection, DEFormer effectively improves the performance of the detector, bringing 2.1% and 3.4% improvement in ExDark and DARK FACE datasets on mAP respectively.

{{</citation>}}


### (17/94) CCSPNet-Joint: Efficient Joint Training Method for Traffic Sign Detection Under Extreme Conditions (Haoqin Hong et al., 2023)

{{<citation>}}

Haoqin Hong, Yue Zhou, Xiangyu Shu, Xiangfang Hu. (2023)  
**CCSPNet-Joint: Efficient Joint Training Method for Traffic Sign Detection Under Extreme Conditions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.06902v2)  

---


**ABSTRACT**  
Traffic sign detection is an important research direction in intelligent driving. Unfortunately, existing methods often overlook extreme conditions such as fog, rain, and motion blur. Moreover, the end-to-end training strategy for image denoising and object detection models fails to utilize inter-model information effectively. To address these issues, we propose CCSPNet, an efficient feature extraction module based on Transformers and CNNs, which effectively leverages contextual information, achieves faster inference speed and provides stronger feature enhancement capabilities. Furthermore, we establish the correlation between object detection and image denoising tasks and propose a joint training model, CCSPNet-Joint, to improve data efficiency and generalization. Finally, to validate our approach, we create the CCTSDB-AUG dataset for traffic sign detection in extreme scenarios. Extensive experiments have shown that CCSPNet achieves state-of-the-art performance in traffic sign detection under extreme conditions. Compared to end-to-end methods, CCSPNet-Joint achieves a 5.32% improvement in precision and an 18.09% improvement in mAP@.5.

{{</citation>}}


### (18/94) MagiCapture: High-Resolution Multi-Concept Portrait Customization (Junha Hyung et al., 2023)

{{<citation>}}

Junha Hyung, Jaeyo Shin, Jaegul Choo. (2023)  
**MagiCapture: High-Resolution Multi-Concept Portrait Customization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.06895v1)  

---


**ABSTRACT**  
Large-scale text-to-image models including Stable Diffusion are capable of generating high-fidelity photorealistic portrait images. There is an active research area dedicated to personalizing these models, aiming to synthesize specific subjects or styles using provided sets of reference images. However, despite the plausible results from these personalization methods, they tend to produce images that often fall short of realism and are not yet on a commercially viable level. This is particularly noticeable in portrait image generation, where any unnatural artifact in human faces is easily discernible due to our inherent human bias. To address this, we introduce MagiCapture, a personalization method for integrating subject and style concepts to generate high-resolution portrait images using just a few subject and style references. For instance, given a handful of random selfies, our fine-tuned model can generate high-quality portrait images in specific styles, such as passport or profile photos. The main challenge with this task is the absence of ground truth for the composed concepts, leading to a reduction in the quality of the final output and an identity shift of the source subject. To address these issues, we present a novel Attention Refocusing loss coupled with auxiliary priors, both of which facilitate robust learning within this weakly supervised learning setting. Our pipeline also includes additional post-processing steps to ensure the creation of highly realistic outputs. MagiCapture outperforms other baselines in both quantitative and qualitative evaluations and can also be generalized to other non-human objects.

{{</citation>}}


### (19/94) Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit? (Bill Psomas et al., 2023)

{{<citation>}}

Bill Psomas, Ioannis Kakogeorgiou, Konstantinos Karantzalos, Yannis Avrithis. (2023)  
**Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.06891v1)  

---


**ABSTRACT**  
Convolutional networks and vision transformers have different forms of pairwise interactions, pooling across layers and pooling at the end of the network. Does the latter really need to be different? As a by-product of pooling, vision transformers provide spatial attention for free, but this is most often of low quality unless self-supervised, which is not well studied. Is supervision really the problem?   In this work, we develop a generic pooling framework and then we formulate a number of existing methods as instantiations. By discussing the properties of each group of methods, we derive SimPool, a simple attention-based pooling mechanism as a replacement of the default one for both convolutional and transformer encoders. We find that, whether supervised or self-supervised, this improves performance on pre-training and downstream tasks and provides attention maps delineating object boundaries in all cases. One could thus call SimPool universal. To our knowledge, we are the first to obtain attention maps in supervised transformers of at least as good quality as self-supervised, without explicit losses or modifying the architecture. Code at: https://github.com/billpsomas/simpool.

{{</citation>}}


### (20/94) SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation (Xian Lin et al., 2023)

{{<citation>}}

Xian Lin, Yangyang Xiang, Li Zhang, Xin Yang, Zengqiang Yan, Li Yu. (2023)  
**SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2309.06824v1)  

---


**ABSTRACT**  
Segment anything model (SAM), an eminent universal image segmentation model, has recently gathered considerable attention within the domain of medical image segmentation. Despite the remarkable performance of SAM on natural images, it grapples with significant performance degradation and limited generalization when confronted with medical images, particularly with those involving objects of low contrast, faint boundaries, intricate shapes, and diminutive sizes. In this paper, we propose SAMUS, a universal model tailored for ultrasound image segmentation. In contrast to previous SAM-based universal models, SAMUS pursues not only better generalization but also lower deployment cost, rendering it more suitable for clinical applications. Specifically, based on SAM, a parallel CNN branch is introduced to inject local features into the ViT encoder through cross-branch attention for better medical image segmentation. Then, a position adapter and a feature adapter are developed to adapt SAM from natural to medical domains and from requiring large-size inputs (1024x1024) to small-size inputs (256x256) for more clinical-friendly deployment. A comprehensive ultrasound dataset, comprising about 30k images and 69k masks and covering six object categories, is collected for verification. Extensive comparison experiments demonstrate SAMUS's superiority against the state-of-the-art task-specific models and universal foundation models under both task-specific evaluation and generalization evaluation. Moreover, SAMUS is deployable on entry-level GPUs, as it has been liberated from the constraints of long sequence encoding. The code, data, and models will be released at https://github.com/xianlin7/SAMUS.

{{</citation>}}


### (21/94) TAP: Targeted Prompting for Task Adaptive Generation of Textual Training Instances for Visual Classification (M. Jehanzeb Mirza et al., 2023)

{{<citation>}}

M. Jehanzeb Mirza, Leonid Karlinsky, Wei Lin, Horst Possegger, Rogerio Feris, Horst Bischof. (2023)  
**TAP: Targeted Prompting for Task Adaptive Generation of Textual Training Instances for Visual Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.06809v1)  

---


**ABSTRACT**  
Vision and Language Models (VLMs), such as CLIP, have enabled visual recognition of a potentially unlimited set of categories described by text prompts. However, for the best visual recognition performance, these models still require tuning to better fit the data distributions of the downstream tasks, in order to overcome the domain shift from the web-based pre-training data. Recently, it has been shown that it is possible to effectively tune VLMs without any paired data, and in particular to effectively improve VLMs visual recognition performance using text-only training data generated by Large Language Models (LLMs). In this paper, we dive deeper into this exciting text-only VLM training approach and explore ways it can be significantly further improved taking the specifics of the downstream task into account when sampling text data from LLMs. In particular, compared to the SOTA text-only VLM training approach, we demonstrate up to 8.4% performance improvement in (cross) domain-specific adaptation, up to 8.7% improvement in fine-grained recognition, and 3.1% overall average improvement in zero-shot classification compared to strong baselines.

{{</citation>}}


### (22/94) Motion-Bias-Free Feature-Based SLAM (Alejandro Fontan et al., 2023)

{{<citation>}}

Alejandro Fontan, Javier Civera, Michael Milford. (2023)  
**Motion-Bias-Free Feature-Based SLAM**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.06792v1)  

---


**ABSTRACT**  
For SLAM to be safely deployed in unstructured real world environments, it must possess several key properties that are not encompassed by conventional benchmarks. In this paper we show that SLAM commutativity, that is, consistency in trajectory estimates on forward and reverse traverses of the same route, is a significant issue for the state of the art. Current pipelines show a significant bias between forward and reverse directions of travel, that is in addition inconsistent regarding which direction of travel exhibits better performance. In this paper we propose several contributions to feature-based SLAM pipelines that remedies the motion bias problem. In a comprehensive evaluation across four datasets, we show that our contributions implemented in ORB-SLAM2 substantially reduce the bias between forward and backward motion and additionally improve the aggregated trajectory error. Removing the SLAM motion bias has significant relevance for the wide range of robotics and computer vision applications where performance consistency is important.

{{</citation>}}


### (23/94) Remote Sensing Object Detection Meets Deep Learning: A Meta-review of Challenges and Advances (Xiangrong Zhang et al., 2023)

{{<citation>}}

Xiangrong Zhang, Tianyang Zhang, Guanchun Wang, Peng Zhu, Xu Tang, Xiuping Jia, Licheng Jiao. (2023)  
**Remote Sensing Object Detection Meets Deep Learning: A Meta-review of Challenges and Advances**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.06751v1)  

---


**ABSTRACT**  
Remote sensing object detection (RSOD), one of the most fundamental and challenging tasks in the remote sensing field, has received longstanding attention. In recent years, deep learning techniques have demonstrated robust feature representation capabilities and led to a big leap in the development of RSOD techniques. In this era of rapid technical evolution, this review aims to present a comprehensive review of the recent achievements in deep learning based RSOD methods. More than 300 papers are covered in this review. We identify five main challenges in RSOD, including multi-scale object detection, rotated object detection, weak object detection, tiny object detection, and object detection with limited supervision, and systematically review the corresponding methods developed in a hierarchical division manner. We also review the widely used benchmark datasets and evaluation metrics within the field of RSOD, as well as the application scenarios for RSOD. Future research directions are provided for further promoting the research in RSOD.

{{</citation>}}


### (24/94) MFL-YOLO: An Object Detection Model for Damaged Traffic Signs (Tengyang Chen et al., 2023)

{{<citation>}}

Tengyang Chen, Jiangtao Ren. (2023)  
**MFL-YOLO: An Object Detection Model for Damaged Traffic Signs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.06750v1)  

---


**ABSTRACT**  
Traffic signs are important facilities to ensure traffic safety and smooth flow, but may be damaged due to many reasons, which poses a great safety hazard. Therefore, it is important to study a method to detect damaged traffic signs. Existing object detection techniques for damaged traffic signs are still absent. Since damaged traffic signs are closer in appearance to normal ones, it is difficult to capture the detailed local damage features of damaged traffic signs using traditional object detection methods. In this paper, we propose an improved object detection method based on YOLOv5s, namely MFL-YOLO (Mutual Feature Levels Loss enhanced YOLO). We designed a simple cross-level loss function so that each level of the model has its own role, which is beneficial for the model to be able to learn more diverse features and improve the fine granularity. The method can be applied as a plug-and-play module and it does not increase the structural complexity or the computational complexity while improving the accuracy. We also replaced the traditional convolution and CSP with the GSConv and VoVGSCSP in the neck of YOLOv5s to reduce the scale and computational complexity. Compared with YOLOv5s, our MFL-YOLO improves 4.3 and 5.1 in F1 scores and mAP, while reducing the FLOPs by 8.9%. The Grad-CAM heat map visualization shows that our model can better focus on the local details of the damaged traffic signs. In addition, we also conducted experiments on CCTSDB2021 and TT100K to further validate the generalization of our model.

{{</citation>}}


### (25/94) Dynamic Spectrum Mixer for Visual Recognition (Zhiqiang Hu et al., 2023)

{{<citation>}}

Zhiqiang Hu, Tao Yu. (2023)  
**Dynamic Spectrum Mixer for Visual Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2309.06721v1)  

---


**ABSTRACT**  
Recently, MLP-based vision backbones have achieved promising performance in several visual recognition tasks. However, the existing MLP-based methods directly aggregate tokens with static weights, leaving the adaptability to different images untouched. Moreover, Recent research demonstrates that MLP-Transformer is great at creating long-range dependencies but ineffective at catching high frequencies that primarily transmit local information, which prevents it from applying to the downstream dense prediction tasks, such as semantic segmentation. To address these challenges, we propose a content-adaptive yet computationally efficient structure, dubbed Dynamic Spectrum Mixer (DSM). The DSM represents token interactions in the frequency domain by employing the Discrete Cosine Transform, which can learn long-term spatial dependencies with log-linear complexity. Furthermore, a dynamic spectrum weight generation layer is proposed as the spectrum bands selector, which could emphasize the informative frequency bands while diminishing others. To this end, the technique can efficiently learn detailed features from visual input that contains both high- and low-frequency information. Extensive experiments show that DSM is a powerful and adaptable backbone for a range of visual recognition tasks. Particularly, DSM outperforms previous transformer-based and MLP-based models, on image classification, object detection, and semantic segmentation tasks, such as 83.8 \% top-1 accuracy on ImageNet, and 49.9 \% mIoU on ADE20K.

{{</citation>}}


### (26/94) STUPD: A Synthetic Dataset for Spatial and Temporal Relation Reasoning (Palaash Agrawal et al., 2023)

{{<citation>}}

Palaash Agrawal, Haidi Azaman, Cheston Tan. (2023)  
**STUPD: A Synthetic Dataset for Spatial and Temporal Relation Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.06680v1)  

---


**ABSTRACT**  
Understanding relations between objects is crucial for understanding the semantics of a visual scene. It is also an essential step in order to bridge visual and language models. However, current state-of-the-art computer vision models still lack the ability to perform spatial reasoning well. Existing datasets mostly cover a relatively small number of spatial relations, all of which are static relations that do not intrinsically involve motion. In this paper, we propose the Spatial and Temporal Understanding of Prepositions Dataset (STUPD) -- a large-scale video dataset for understanding static and dynamic spatial relationships derived from prepositions of the English language. The dataset contains 150K visual depictions (videos and images), consisting of 30 distinct spatial prepositional senses, in the form of object interaction simulations generated synthetically using Unity3D. In addition to spatial relations, we also propose 50K visual depictions across 10 temporal relations, consisting of videos depicting event/time-point interactions. To our knowledge, no dataset exists that represents temporal relations through visual settings. In this dataset, we also provide 3D information about object interactions such as frame-wise coordinates, and descriptions of the objects used. The goal of this synthetic dataset is to help models perform better in visual relationship detection in real-world settings. We demonstrate an increase in the performance of various models over 2 real-world datasets (ImageNet-VidVRD and Spatial Senses) when pretrained on the STUPD dataset, in comparison to other pretraining datasets.

{{</citation>}}


### (27/94) ShaDocFormer: A Shadow-attentive Threshold Detector with Cascaded Fusion Refiner for document shadow removal (Weiwen Chen et al., 2023)

{{<citation>}}

Weiwen Chen, Shenghong Luo, Xuhang Chen, Zinuo Li, Shuqiang Wang, Chi-Man Pun. (2023)  
**ShaDocFormer: A Shadow-attentive Threshold Detector with Cascaded Fusion Refiner for document shadow removal**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.06670v2)  

---


**ABSTRACT**  
Document shadow is a common issue that arise when capturing documents using mobile devices, which significantly impacts the readability. Current methods encounter various challenges including inaccurate detection of shadow masks and estimation of illumination. In this paper, we propose ShaDocFormer, a Transformer-based architecture that integrates traditional methodologies and deep learning techniques to tackle the problem of document shadow removal. The ShaDocFormer architecture comprises two components: the Shadow-attentive Threshold Detector (STD) and the Cascaded Fusion Refiner (CFR). The STD module employs a traditional thresholding technique and leverages the attention mechanism of the Transformer to gather global information, thereby enabling precise detection of shadow masks. The cascaded and aggregative structure of the CFR module facilitates a coarse-to-fine restoration process for the entire image. As a result, ShaDocFormer excels in accurately detecting and capturing variations in both shadow and illumination, thereby enabling effective removal of shadows. Extensive experiments demonstrate that ShaDocFormer outperforms current state-of-the-art methods in both qualitative and quantitative measurements.

{{</citation>}}


### (28/94) LCReg: Long-Tailed Image Classification with Latent Categories based Recognition (Weide Liu et al., 2023)

{{<citation>}}

Weide Liu, Zhonghua Wu, Yiming Wang, Henghui Ding, Fayao Liu, Jie Lin, Guosheng Lin. (2023)  
**LCReg: Long-Tailed Image Classification with Latent Categories based Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2309.07186v1)  

---


**ABSTRACT**  
In this work, we tackle the challenging problem of long-tailed image recognition. Previous long-tailed recognition approaches mainly focus on data augmentation or re-balancing strategies for the tail classes to give them more attention during model training. However, these methods are limited by the small number of training images for the tail classes, which results in poor feature representations. To address this issue, we propose the Latent Categories based long-tail Recognition (LCReg) method. Our hypothesis is that common latent features shared by head and tail classes can be used to improve feature representation. Specifically, we learn a set of class-agnostic latent features shared by both head and tail classes, and then use semantic data augmentation on the latent features to implicitly increase the diversity of the training sample. We conduct extensive experiments on five long-tailed image recognition datasets, and the results show that our proposed method significantly improves the baselines.

{{</citation>}}


## cs.NI (2)



### (29/94) A Simple Non-Deterministic Approach Can Adapt to Complex Unpredictable 5G Cellular Networks (Parsa Pazhooheshy et al., 2023)

{{<citation>}}

Parsa Pazhooheshy, Soheil Abbasloo, Yashar Ganjali. (2023)  
**A Simple Non-Deterministic Approach Can Adapt to Complex Unpredictable 5G Cellular Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.07324v1)  

---


**ABSTRACT**  
5G cellular networks are envisioned to support a wide range of emerging delay-oriented services with different delay requirements (e.g., 20ms for VR/AR, 40ms for cloud gaming, and 100ms for immersive video streaming). However, due to the highly variable and unpredictable nature of 5G access links, existing end-to-end (e2e) congestion control (CC) schemes perform poorly for them. In this paper, we demonstrate that properly blending non-deterministic exploration techniques with straightforward proactive and reactive measures is sufficient to design a simple yet effective e2e CC scheme for 5G networks that can: (1) achieve high controllable performance, and (2) possess provable properties. To that end, we designed Reminis and through extensive experiments on emulated and real-world 5G networks, show the performance benefits of it compared with different CC schemes. For instance, averaged over 60 different 5G cellular links on the Standalone (SA) scenarios, compared with a recent design by Google (BBR2), Reminis can achieve 2.2x lower 95th percentile delay while having the same link utilization.

{{</citation>}}


### (30/94) Safe and Accelerated Deep Reinforcement Learning-based O-RAN Slicing: A Hybrid Transfer Learning Approach (Ahmad M. Nagib et al., 2023)

{{<citation>}}

Ahmad M. Nagib, Hatem Abou-Zeid, Hossam S. Hassanein. (2023)  
**Safe and Accelerated Deep Reinforcement Learning-based O-RAN Slicing: A Hybrid Transfer Learning Approach**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.07265v1)  

---


**ABSTRACT**  
The open radio access network (O-RAN) architecture supports intelligent network control algorithms as one of its core capabilities. Data-driven applications incorporate such algorithms to optimize radio access network (RAN) functions via RAN intelligent controllers (RICs). Deep reinforcement learning (DRL) algorithms are among the main approaches adopted in the O-RAN literature to solve dynamic radio resource management problems. However, despite the benefits introduced by the O-RAN RICs, the practical adoption of DRL algorithms in real network deployments falls behind. This is primarily due to the slow convergence and unstable performance exhibited by DRL agents upon deployment and when facing previously unseen network conditions. In this paper, we address these challenges by proposing transfer learning (TL) as a core component of the training and deployment workflows for the DRL-based closed-loop control of O-RAN functionalities. To this end, we propose and design a hybrid TL-aided approach that leverages the advantages of both policy reuse and distillation TL methods to provide safe and accelerated convergence in DRL-based O-RAN slicing. We conduct a thorough experiment that accommodates multiple services, including real VR gaming traffic to reflect practical scenarios of O-RAN slicing. We also propose and implement policy reuse and distillation-aided DRL and non-TL-aided DRL as three separate baselines. The proposed hybrid approach shows at least: 7.7% and 20.7% improvements in the average initial reward value and the percentage of converged scenarios, and a 64.6% decrease in reward variance while maintaining fast convergence and enhancing the generalizability compared with the baselines.

{{</citation>}}


## cs.CL (22)



### (31/94) Traveling Words: A Geometric Interpretation of Transformers (Raul Molina, 2023)

{{<citation>}}

Raul Molina. (2023)  
**Traveling Words: A Geometric Interpretation of Transformers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.07315v1)  

---


**ABSTRACT**  
Transformers have significantly advanced the field of natural language processing, but comprehending their internal mechanisms remains a challenge. In this paper, we introduce a novel geometric perspective that elucidates the inner mechanisms of transformer operations. Our primary contribution is illustrating how layer normalization confines the latent features to a hyper-sphere, subsequently enabling attention to mold the semantic representation of words on this surface. This geometric viewpoint seamlessly connects established properties such as iterative refinement and contextual embeddings. We validate our insights by probing a pre-trained 124M parameter GPT-2 model. Our findings reveal clear query-key attention patterns in early layers and build upon prior observations regarding the subject-specific nature of attention heads at deeper layers. Harnessing these geometric insights, we present an intuitive understanding of transformers, depicting them as processes that model the trajectory of word particles along the hyper-sphere.

{{</citation>}}


### (32/94) Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs (Angelica Chen et al., 2023)

{{<citation>}}

Angelica Chen, Ravid Schwartz-Ziv, Kyunghyun Cho, Matthew L. Leavitt, Naomi Saphra. (2023)  
**Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Bias, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2309.07311v1)  

---


**ABSTRACT**  
Most interpretability research in NLP focuses on understanding the behavior and features of a fully trained model. However, certain insights into model behavior may only be accessible by observing the trajectory of the training process. In this paper, we present a case study of syntax acquisition in masked language models (MLMs). Our findings demonstrate how analyzing the evolution of interpretable artifacts throughout training deepens our understanding of emergent behavior. In particular, we study Syntactic Attention Structure (SAS), a naturally emerging property of MLMs wherein specific Transformer heads tend to focus on specific syntactic relations. We identify a brief window in training when models abruptly acquire SAS and find that this window is concurrent with a steep drop in loss. Moreover, SAS precipitates the subsequent acquisition of linguistic capabilities. We then examine the causal role of SAS by introducing a regularizer to manipulate SAS during training, and demonstrate that SAS is necessary for the development of grammatical capabilities. We further find that SAS competes with other beneficial traits and capabilities during training, and that briefly suppressing SAS can improve model quality. These findings reveal a real-world example of the relationship between disadvantageous simplicity bias and interpretable breakthrough training dynamics.

{{</citation>}}


### (33/94) In-Contextual Bias Suppression for Large Language Models (Daisuke Oba et al., 2023)

{{<citation>}}

Daisuke Oba, Masahiro Kaneko, Danushka Bollegala. (2023)  
**In-Contextual Bias Suppression for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.07251v1)  

---


**ABSTRACT**  
Despite their impressive performance in a wide range of NLP tasks, Large Language Models (LLMs) have been reported to encode worrying-levels of gender bias. Prior work has proposed debiasing methods that require human labelled examples, data augmentation and fine-tuning of the LLMs, which are computationally costly. Moreover, one might not even have access to the internal parameters for performing debiasing such as in the case of commercially available LLMs such as GPT-4. To address this challenge we propose bias suppression, a novel alternative to debiasing that does not require access to model parameters. We show that text-based preambles, generated from manually designed templates covering counterfactual statements, can accurately suppress gender biases in LLMs. Moreover, we find that descriptive sentences for occupations can further suppress gender biases. Interestingly, we find that bias suppression has a minimal adverse effect on downstream task performance, while effectively mitigating the gender biases.

{{</citation>}}


### (34/94) RAIN: Your Language Models Can Align Themselves without Finetuning (Yuhui Li et al., 2023)

{{<citation>}}

Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang. (2023)  
**RAIN: Your Language Models Can Align Themselves without Finetuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.07124v1)  

---


**ABSTRACT**  
Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, the so-called finetuning step. In contrast, aligning frozen LLMs without any extra data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide backward rewind and forward generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates; during the self-evaluation phase, the model receives guidance on which human preference to align with through a fixed-template prompt, eliminating the need to modify the initial prompt. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B over vanilla inference from 82% to 97%, while maintaining the helpfulness rate. Under the leading adversarial attack llm-attacks on Vicuna 33B, RAIN establishes a new defense baseline by reducing the attack success rate from 94% to 19%.

{{</citation>}}


### (35/94) Sight Beyond Text: Multi-Modal Training Enhances LLMs in Truthfulness and Ethics (Haoqin Tu et al., 2023)

{{<citation>}}

Haoqin Tu, Bingchen Zhao, Chen Wei, Cihang Xie. (2023)  
**Sight Beyond Text: Multi-Modal Training Enhances LLMs in Truthfulness and Ethics**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-CY, cs-LG, cs.CL  
Keywords: LLaMA, NLP, QA  
[Paper Link](http://arxiv.org/abs/2309.07120v1)  

---


**ABSTRACT**  
Multi-modal large language models (MLLMs) are trained based on large language models (LLM), with an enhanced capability to comprehend multi-modal inputs and generate textual responses. While they excel in multi-modal tasks, the pure NLP abilities of MLLMs are often underestimated and left untested. In this study, we get out of the box and unveil an intriguing characteristic of MLLMs -- our preliminary results suggest that visual instruction tuning, a prevailing strategy for transitioning LLMs into MLLMs, unexpectedly and interestingly helps models attain both improved truthfulness and ethical alignment in the pure NLP context. For example, a visual-instruction-tuned LLaMA2 7B model surpasses the performance of the LLaMA2-chat 7B model, fine-tuned with over one million human annotations, on TruthfulQA-mc and Ethics benchmarks. Further analysis reveals that the improved alignment can be attributed to the superior instruction quality inherent to visual-text data. In releasing our code at github.com/UCSC-VLAA/Sight-Beyond-Text, we aspire to foster further exploration into the intrinsic value of visual-text synergies and, in a broader scope, multi-modal interactions in alignment research.

{{</citation>}}


### (36/94) Mitigating Hallucinations and Off-target Machine Translation with Source-Contrastive and Language-Contrastive Decoding (Rico Sennrich et al., 2023)

{{<citation>}}

Rico Sennrich, Jannis Vamvas, Alireza Mohammadshahi. (2023)  
**Mitigating Hallucinations and Off-target Machine Translation with Source-Contrastive and Language-Contrastive Decoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2309.07098v1)  

---


**ABSTRACT**  
Hallucinations and off-target translation remain unsolved problems in machine translation, especially for low-resource languages and massively multilingual models. In this paper, we introduce methods to mitigate both failure cases with a modified decoding objective, without requiring retraining or external models. In source-contrastive decoding, we search for a translation that is probable given the correct input, but improbable given a random input segment, hypothesising that hallucinations will be similarly probable given either. In language-contrastive decoding, we search for a translation that is probable, but improbable given the wrong language indicator token. In experiments on M2M-100 (418M) and SMaLL-100, we find that these methods effectively suppress hallucinations and off-target translations, improving chrF2 by 1.7 and 1.4 points on average across 57 tested translation directions. In a proof of concept on English--German, we also show that we can suppress off-target translations with the Llama 2 chat models, demonstrating the applicability of the method to machine translation with LLMs. We release our source code at https://github.com/ZurichNLP/ContraDecode.

{{</citation>}}


### (37/94) SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions (Zhexin Zhang et al., 2023)

{{<citation>}}

Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, Minlie Huang. (2023)  
**SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.07045v1)  

---


**ABSTRACT**  
With the rapid development of Large Language Models (LLMs), increasing attention has been paid to their safety concerns. Consequently, evaluating the safety of LLMs has become an essential task for facilitating the broad applications of LLMs. Nevertheless, the absence of comprehensive safety evaluation benchmarks poses a significant impediment to effectively assess and enhance the safety of LLMs. In this work, we present SafetyBench, a comprehensive benchmark for evaluating the safety of LLMs, which comprises 11,435 diverse multiple choice questions spanning across 7 distinct categories of safety concerns. Notably, SafetyBench also incorporates both Chinese and English data, facilitating the evaluation in both languages. Our extensive tests over 25 popular Chinese and English LLMs in both zero-shot and few-shot settings reveal a substantial performance advantage for GPT-4 over its counterparts, and there is still significant room for improving the safety of current LLMs. We believe SafetyBench will enable fast and comprehensive evaluation of LLMs' safety, and foster the development of safer LLMs. Data and evaluation guidelines are available at https://github.com/thu-coai/SafetyBench. Submission entrance and leaderboard are available at https://llmbench.ai/safety.

{{</citation>}}


### (38/94) How (Not) to Use Sociodemographic Information for Subjective NLP Tasks (Tilman Beck et al., 2023)

{{<citation>}}

Tilman Beck, Hendrik Schuff, Anne Lauscher, Iryna Gurevych. (2023)  
**How (Not) to Use Sociodemographic Information for Subjective NLP Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.07034v1)  

---


**ABSTRACT**  
Annotators' sociodemographic backgrounds (i.e., the individual compositions of their gender, age, educational background, etc.) have a strong impact on their decisions when working on subjective NLP tasks, such as hate speech detection. Often, heterogeneous backgrounds result in high disagreements. To model this variation, recent work has explored sociodemographic prompting, a technique, which steers the output of prompt-based models towards answers that humans with specific sociodemographic profiles would give. However, the available NLP literature disagrees on the efficacy of this technique -- it remains unclear, for which tasks and scenarios it can help and evaluations are limited to specific tasks only. We address this research gap by presenting the largest and most comprehensive study of sociodemographic prompting today. Concretely, we evaluate several prompt formulations across seven datasets and six instruction-tuned model families. We find that (1) while sociodemographic prompting can be beneficial for improving zero-shot learning in subjective NLP tasks, (2) its outcomes largely vary for different model types, sizes, and datasets, (3) are subject to large variance with regards to prompt formulations. Thus, sociodemographic prompting is not a reliable proxy for traditional data annotation with a sociodemographically heterogeneous group of annotators. Instead, we propose (4) to use it for identifying ambiguous instances resulting in more informed annotation efforts.

{{</citation>}}


### (39/94) Beyond original Research Articles Categorization via NLP (Rosanna Turrisi, 2023)

{{<citation>}}

Rosanna Turrisi. (2023)  
**Beyond original Research Articles Categorization via NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.07020v1)  

---


**ABSTRACT**  
This work proposes a novel approach to text categorization -- for unknown categories -- in the context of scientific literature, using Natural Language Processing techniques. The study leverages the power of pre-trained language models, specifically SciBERT, to extract meaningful representations of abstracts from the ArXiv dataset. Text categorization is performed using the K-Means algorithm, and the optimal number of clusters is determined based on the Silhouette score. The results demonstrate that the proposed approach captures subject information more effectively than the traditional arXiv labeling system, leading to improved text categorization. The approach offers potential for better navigation and recommendation systems in the rapidly growing landscape of scientific research literature.

{{</citation>}}


### (40/94) OYXOY: A Modern NLP Test Suite for Modern Greek (Konstantinos Kogkalidis et al., 2023)

{{<citation>}}

Konstantinos Kogkalidis, Stergios Chatzikyriakidis, Eirini Chrysovalantou Giannikouri, Vassiliki Katsouli, Christina Klironomou, Christina Koula, Dimitris Papadakis, Thelka Pasparaki, Erofili Psaltaki, Efthymia Sakellariou, Hara Soupiona. (2023)  
**OYXOY: A Modern NLP Test Suite for Modern Greek**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2309.07009v1)  

---


**ABSTRACT**  
This paper serves as a foundational step towards the development of a linguistically motivated and technically relevant evaluation suite for Greek NLP. We initiate this endeavor by introducing four expert-verified evaluation tasks, specifically targeted at natural language inference, word sense disambiguation (through example comparison or sense selection) and metaphor detection. More than language-adapted replicas of existing tasks, we contribute two innovations which will resonate with the broader resource and evaluation community. Firstly, our inference dataset is the first of its kind, marking not just \textit{one}, but rather \textit{all} possible inference labels, accounting for possible shifts due to e.g. ambiguity or polysemy. Secondly, we demonstrate a cost-efficient method to obtain datasets for under-resourced languages. Using ChatGPT as a language-neutral parser, we transform the Dictionary of Standard Modern Greek into a structured format, from which we derive the other three tasks through simple projections. Alongside each task, we conduct experiments using currently available state of the art machinery. Our experimental baselines affirm the challenging nature of our tasks and highlight the need for expedited progress in order for the Greek NLP ecosystem to keep pace with contemporary mainstream research.

{{</citation>}}


### (41/94) Dynamic Causal Disentanglement Model for Dialogue Emotion Detection (Yuting Su et al., 2023)

{{<citation>}}

Yuting Su, Yichen Wei, Weizhi Nie, Sicheng Zhao, Anan Liu. (2023)  
**Dynamic Causal Disentanglement Model for Dialogue Emotion Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, GPT-4, LSTM  
[Paper Link](http://arxiv.org/abs/2309.06928v1)  

---


**ABSTRACT**  
Emotion detection is a critical technology extensively employed in diverse fields. While the incorporation of commonsense knowledge has proven beneficial for existing emotion detection methods, dialogue-based emotion detection encounters numerous difficulties and challenges due to human agency and the variability of dialogue content.In dialogues, human emotions tend to accumulate in bursts. However, they are often implicitly expressed. This implies that many genuine emotions remain concealed within a plethora of unrelated words and dialogues.In this paper, we propose a Dynamic Causal Disentanglement Model based on hidden variable separation, which is founded on the separation of hidden variables. This model effectively decomposes the content of dialogues and investigates the temporal accumulation of emotions, thereby enabling more precise emotion recognition. First, we introduce a novel Causal Directed Acyclic Graph (DAG) to establish the correlation between hidden emotional information and other observed elements. Subsequently, our approach utilizes pre-extracted personal attributes and utterance topics as guiding factors for the distribution of hidden variables, aiming to separate irrelevant ones. Specifically, we propose a dynamic temporal disentanglement model to infer the propagation of utterances and hidden variables, enabling the accumulation of emotion-related information throughout the conversation. To guide this disentanglement process, we leverage the ChatGPT-4.0 and LSTM networks to extract utterance topics and personal attributes as observed information.Finally, we test our approach on two popular datasets in dialogue emotion detection and relevant experimental results verified the model's superiority.

{{</citation>}}


### (42/94) Native Language Identification with Big Bird Embeddings (Sergey Kramp et al., 2023)

{{<citation>}}

Sergey Kramp, Giovanni Cassani, Chris Emmery. (2023)  
**Native Language Identification with Big Bird Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Language Identification, NLI  
[Paper Link](http://arxiv.org/abs/2309.06923v1)  

---


**ABSTRACT**  
Native Language Identification (NLI) intends to classify an author's native language based on their writing in another language. Historically, the task has heavily relied on time-consuming linguistic feature engineering, and transformer-based NLI models have thus far failed to offer effective, practical alternatives. The current work investigates if input size is a limiting factor, and shows that classifiers trained using Big Bird embeddings outperform linguistic feature engineering models by a large margin on the Reddit-L2 dataset. Additionally, we provide further insight into input length dependencies, show consistent out-of-sample performance, and qualitatively analyze the embedding space. Given the effectiveness and computational efficiency of this method, we believe it offers a promising avenue for future NLI work.

{{</citation>}}


### (43/94) Continual Learning with Dirichlet Generative-based Rehearsal (Min Zeng et al., 2023)

{{<citation>}}

Min Zeng, Wei Xue, Qifeng Liu, Yike Guo. (2023)  
**Continual Learning with Dirichlet Generative-based Rehearsal**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.06917v1)  

---


**ABSTRACT**  
Recent advancements in data-driven task-oriented dialogue systems (ToDs) struggle with incremental learning due to computational constraints and time-consuming issues. Continual Learning (CL) attempts to solve this by avoiding intensive pre-training, but it faces the problem of catastrophic forgetting (CF). While generative-based rehearsal CL methods have made significant strides, generating pseudo samples that accurately reflect the underlying task-specific distribution is still a challenge. In this paper, we present Dirichlet Continual Learning (DCL), a novel generative-based rehearsal strategy for CL. Unlike the traditionally used Gaussian latent variable in the Conditional Variational Autoencoder (CVAE), DCL leverages the flexibility and versatility of the Dirichlet distribution to model the latent prior variable. This enables it to efficiently capture sentence-level features of previous tasks and effectively guide the generation of pseudo samples. In addition, we introduce Jensen-Shannon Knowledge Distillation (JSKD), a robust logit-based knowledge distillation method that enhances knowledge transfer during pseudo sample generation. Our experiments confirm the efficacy of our approach in both intent detection and slot-filling tasks, outperforming state-of-the-art methods.

{{</citation>}}


### (44/94) Towards the TopMost: A Topic Modeling System Toolkit (Xiaobao Wu et al., 2023)

{{<citation>}}

Xiaobao Wu, Fengjun Pan, Anh Tuan Luu. (2023)  
**Towards the TopMost: A Topic Modeling System Toolkit**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2309.06908v1)  

---


**ABSTRACT**  
Topic models have been proposed for decades with various applications and recently refreshed by the neural variational inference. However, these topic models adopt totally distinct dataset, implementation, and evaluation settings, which hinders their quick utilization and fair comparisons. This greatly hinders the research progress of topic models. To address these issues, in this paper we propose a Topic Modeling System Toolkit (TopMost). Compared to existing toolkits, TopMost stands out by covering a wider range of topic modeling scenarios including complete lifecycles with dataset pre-processing, model training, testing, and evaluations. The highly cohesive and decoupled modular design of TopMost enables quick utilization, fair comparisons, and flexible extensions of different topic models. This can facilitate the research and applications of topic models. Our code, tutorials, and documentation are available at https://github.com/bobxwu/topmost.

{{</citation>}}


### (45/94) Comparative Analysis of Contextual Relation Extraction based on Deep Learning Models (R. Priyadharshini et al., 2023)

{{<citation>}}

R. Priyadharshini, G. Jeyakodi, P. Shanthi Bala. (2023)  
**Comparative Analysis of Contextual Relation Extraction based on Deep Learning Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, Natural Language Processing, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2309.06814v1)  

---


**ABSTRACT**  
Contextual Relation Extraction (CRE) is mainly used for constructing a knowledge graph with a help of ontology. It performs various tasks such as semantic search, query answering, and textual entailment. Relation extraction identifies the entities from raw texts and the relations among them. An efficient and accurate CRE system is essential for creating domain knowledge in the biomedical industry. Existing Machine Learning and Natural Language Processing (NLP) techniques are not suitable to predict complex relations from sentences that consist of more than two relations and unspecified entities efficiently. In this work, deep learning techniques have been used to identify the appropriate semantic relation based on the context from multiple sentences. Even though various machine learning models have been used for relation extraction, they provide better results only for binary relations, i.e., relations occurred exactly between the two entities in a sentence. Machine learning models are not suited for complex sentences that consist of the words that have various meanings. To address these issues, hybrid deep learning models have been used to extract the relations from complex sentence effectively. This paper explores the analysis of various deep learning models that are used for relation extraction.

{{</citation>}}


### (46/94) Cognitive Mirage: A Review of Hallucinations in Large Language Models (Hongbin Ye et al., 2023)

{{<citation>}}

Hongbin Ye, Tong Liu, Aijia Zhang, Wei Hua, Weiqiang Jia. (2023)  
**Cognitive Mirage: A Review of Hallucinations in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.06794v1)  

---


**ABSTRACT**  
As large language models continue to develop in the field of AI, text generation systems are susceptible to a worrisome phenomenon known as hallucination. In this study, we summarize recent compelling insights into hallucinations in LLMs. We present a novel taxonomy of hallucinations from various text generation tasks, thus provide theoretical insights, detection methods and improvement approaches. Based on this, future research directions are proposed. Our contribution are threefold: (1) We provide a detailed and complete taxonomy for hallucinations appearing in text generation tasks; (2) We provide theoretical analyses of hallucinations in LLMs and provide existing detection and improvement methods; (3) We propose several research directions that can be developed in the future. As hallucinations garner significant attention from the community, we will maintain updates on relevant research progress.

{{</citation>}}


### (47/94) Scaled Prompt-Tuning for Few-Shot Natural Language Generation (Ting Hu et al., 2023)

{{<citation>}}

Ting Hu, Christoph Meinel, Haojin Yang. (2023)  
**Scaled Prompt-Tuning for Few-Shot Natural Language Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2309.06759v1)  

---


**ABSTRACT**  
The increasingly Large Language Models (LLMs) demonstrate stronger language understanding and generation capabilities, while the memory demand and computation cost of fine-tuning LLMs on downstream tasks are non-negligible. Besides, fine-tuning generally requires a certain amount of data from individual tasks whilst data collection cost is another issue to consider in real-world applications. In this work, we focus on Parameter-Efficient Fine-Tuning (PEFT) methods for few-shot Natural Language Generation (NLG), which freeze most parameters in LLMs and tune a small subset of parameters in few-shot cases so that memory footprint, training cost, and labeling cost are reduced while maintaining or even improving the performance. We propose a Scaled Prompt-Tuning (SPT) method which surpasses conventional PT with better performance and generalization ability but without an obvious increase in training cost. Further study on intermediate SPT suggests the superior transferability of SPT in few-shot scenarios, providing a recipe for data-deficient and computation-limited circumstances. Moreover, a comprehensive comparison of existing PEFT methods reveals that certain approaches exhibiting decent performance with modest training cost such as Prefix-Tuning in prior study could struggle in few-shot NLG tasks, especially on challenging datasets.

{{</citation>}}


### (48/94) CONVERSER: Few-Shot Conversational Dense Retrieval with Synthetic Data Generation (Chao-Wei Huang et al., 2023)

{{<citation>}}

Chao-Wei Huang, Chen-Yu Hsu, Tsu-Yuan Hsu, Chen-An Li, Yun-Nung Chen. (2023)  
**CONVERSER: Few-Shot Conversational Dense Retrieval with Synthetic Data Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.06748v1)  

---


**ABSTRACT**  
Conversational search provides a natural interface for information retrieval (IR). Recent approaches have demonstrated promising results in applying dense retrieval to conversational IR. However, training dense retrievers requires large amounts of in-domain paired data. This hinders the development of conversational dense retrievers, as abundant in-domain conversations are expensive to collect. In this paper, we propose CONVERSER, a framework for training conversational dense retrievers with at most 6 examples of in-domain dialogues. Specifically, we utilize the in-context learning capability of large language models to generate conversational queries given a passage in the retrieval corpus. Experimental results on conversational retrieval benchmarks OR-QuAC and TREC CAsT 19 show that the proposed CONVERSER achieves comparable performance to fully-supervised models, demonstrating the effectiveness of our proposed framework in few-shot conversational dense retrieval. All source code and generated datasets are available at https://github.com/MiuLab/CONVERSER

{{</citation>}}


### (49/94) Simultaneous Machine Translation with Large Language Models (Minghan Wang et al., 2023)

{{<citation>}}

Minghan Wang, Jinming Zhao, Thuy-Trang Vu, Fatemeh Shiri, Ehsan Shareghi, Gholamreza Haffari. (2023)  
**Simultaneous Machine Translation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.06706v1)  

---


**ABSTRACT**  
Large language models (LLM) have demonstrated their abilities to solve various natural language processing tasks through dialogue-based interactions. For instance, research indicates that LLMs can achieve competitive performance in offline machine translation tasks for high-resource languages. However, applying LLMs to simultaneous machine translation (SimulMT) poses many challenges, including issues related to the training-inference mismatch arising from different decoding patterns. In this paper, we explore the feasibility of utilizing LLMs for SimulMT. Building upon conventional approaches, we introduce a simple yet effective mixture policy that enables LLMs to engage in SimulMT without requiring additional training. Furthermore, after Supervised Fine-Tuning (SFT) on a mixture of full and prefix sentences, the model exhibits significant performance improvements. Our experiments, conducted with Llama2-7B-chat on nine language pairs from the MUST-C dataset, demonstrate that LLM can achieve translation quality and latency comparable to dedicated SimulMT models.

{{</citation>}}


### (50/94) Benchmarking Procedural Language Understanding for Low-Resource Languages: A Case Study on Turkish (Arda Uzunoğlu et al., 2023)

{{<citation>}}

Arda Uzunoğlu, Gözde Gül Şahin. (2023)  
**Benchmarking Procedural Language Understanding for Low-Resource Languages: A Case Study on Turkish**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Low-Resource, T5  
[Paper Link](http://arxiv.org/abs/2309.06698v1)  

---


**ABSTRACT**  
Understanding procedural natural language (e.g., step-by-step instructions) is a crucial step to execution and planning. However, while there are ample corpora and downstream tasks available in English, the field lacks such resources for most languages. To address this gap, we conduct a case study on Turkish procedural texts. We first expand the number of tutorials in Turkish wikiHow from 2,000 to 52,000 using automated translation tools, where the translation quality and loyalty to the original meaning are validated by a team of experts on a random set. Then, we generate several downstream tasks on the corpus, such as linking actions, goal inference, and summarization. To tackle these tasks, we implement strong baseline models via fine-tuning large language-specific models such as TR-BART and BERTurk, as well as multilingual models such as mBART, mT5, and XLM. We find that language-specific models consistently outperform their multilingual models by a significant margin across most procedural language understanding (PLU) tasks. We release our corpus, downstream tasks and the baseline models with https://github.com/ GGLAB-KU/turkish-plu.

{{</citation>}}


### (51/94) Offline Prompt Evaluation and Optimization with Inverse Reinforcement Learning (Hao Sun, 2023)

{{<citation>}}

Hao Sun. (2023)  
**Offline Prompt Evaluation and Optimization with Inverse Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06553v1)  

---


**ABSTRACT**  
The recent advances in the development of Large Language Models (LLMs) like ChatGPT have achieved remarkable performance by leveraging human expertise. Yet, fully eliciting LLMs' potential for complex tasks requires navigating the vast search space of natural language prompts. While prompt engineering has shown promise, the requisite human-crafted prompts in trial-and-error attempts and the associated costs pose significant challenges. Crucially, the efficiency of prompt optimization hinges on the costly procedure of prompt evaluation. This work introduces Prompt-OIRL, an approach rooted in offline inverse reinforcement learning that seeks to bridge the gap between effective prompt evaluation and affordability. Our method draws on offline datasets from expert evaluations, employing Inverse-RL to derive a reward model for offline, query-dependent prompt evaluations. The advantages of Prompt-OIRL are manifold: it predicts prompt performance, is cost-efficient, produces human-readable results, and efficiently navigates the prompt space. We validate our method across four LLMs and three arithmetic datasets, highlighting its potential as a robust and effective tool for offline prompt evaluation and optimization. Our code as well as the offline datasets are released, and we highlight the Prompt-OIRL can be reproduced within a few hours using a single laptop using CPU

{{</citation>}}


### (52/94) Statistical Rejection Sampling Improves Preference Optimization (Tianqi Liu et al., 2023)

{{<citation>}}

Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J. Liu, Jialu Liu. (2023)  
**Statistical Rejection Sampling Improves Preference Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06657v1)  

---


**ABSTRACT**  
Improving the alignment of language models with human preferences remains an active research challenge. Previous approaches have primarily utilized Reinforcement Learning from Human Feedback (RLHF) via online RL methods such as Proximal Policy Optimization (PPO). Recently, offline methods such as Sequence Likelihood Calibration (SLiC) and Direct Preference Optimization (DPO) have emerged as attractive alternatives, offering improvements in stability and scalability while maintaining competitive performance. SLiC refines its loss function using sequence pairs sampled from a supervised fine-tuned (SFT) policy, while DPO directly optimizes language models based on preference data, foregoing the need for a separate reward model. However, the maximum likelihood estimator (MLE) of the target optimal policy requires labeled preference pairs sampled from that policy. DPO's lack of a reward model constrains its ability to sample preference pairs from the optimal policy, and SLiC is restricted to sampling preference pairs only from the SFT policy. To address these limitations, we introduce a novel approach called Statistical Rejection Sampling Optimization (RSO) that aims to source preference data from the target optimal policy using rejection sampling, enabling a more accurate estimation of the optimal policy. We also propose a unified framework that enhances the loss functions used in both SLiC and DPO from a preference modeling standpoint. Through extensive experiments across three diverse tasks, we demonstrate that RSO consistently outperforms both SLiC and DPO on evaluations from both Large Language Model (LLM) and human raters.

{{</citation>}}


## cs.HC (5)



### (53/94) User Training with Error Augmentation for Electromyogram-based Gesture Classification (Yunus Bicer et al., 2023)

{{<citation>}}

Yunus Bicer, Niklas Smedemark-Margulies, Basak Celik, Elifnur Sunger, Ryan Orendorff, Stephanie Naufel, Tales Imbiriba, Deniz Erdo{ğ}mu{ş}, Eugene Tunik, Mathew Yarossi. (2023)  
**User Training with Error Augmentation for Electromyogram-based Gesture Classification**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC, eess-SP  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.07289v1)  

---


**ABSTRACT**  
We designed and tested a system for real-time control of a user interface by extracting surface electromyographic (sEMG) activity from eight electrodes in a wrist-band configuration. sEMG data were streamed into a machine-learning algorithm that classified hand gestures in real-time. After an initial model calibration, participants were presented with one of three types of feedback during a human-learning stage: veridical feedback, in which predicted probabilities from the gesture classification algorithm were displayed without alteration, modified feedback, in which we applied a hidden augmentation of error to these probabilities, and no feedback. User performance was then evaluated in a series of minigames, in which subjects were required to use eight gestures to manipulate their game avatar to complete a task. Experimental results indicated that, relative to baseline, the modified feedback condition led to significantly improved accuracy and improved gesture class separation. These findings suggest that real-time feedback in a gamified user interface with manipulation of feedback may enable intuitive, rapid, and accurate task acquisition for sEMG-based gesture recognition applications.

{{</citation>}}


### (54/94) Human-Robot Co-creativity: A Scoping Review -- Informing a Research Agenda for Human-Robot Co-Creativity with Older Adults (Marianne Bossema et al., 2023)

{{<citation>}}

Marianne Bossema, Somaya Ben Allouch, Aske Plaat, Rob Saunders. (2023)  
**Human-Robot Co-creativity: A Scoping Review -- Informing a Research Agenda for Human-Robot Co-Creativity with Older Adults**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.07033v1)  

---


**ABSTRACT**  
This review is the first step in a long-term research project exploring how social robotics and AI-generated content can contribute to the creative experiences of older adults, with a focus on collaborative drawing and painting. We systematically searched and selected literature on human-robot co-creativity, and analyzed articles to identify methods and strategies for researching co-creative robotics. We found that none of the studies involved older adults, which shows the gap in the literature for this often involved participant group in robotics research. The analyzed literature provides valuable insights into the design of human-robot co-creativity and informs a research agenda to further investigate the topic with older adults. We argue that future research should focus on ecological and developmental perspectives on creativity, on how system behavior can be aligned with the values of older adults, and on the system structures that support this best.

{{</citation>}}


### (55/94) Human-Machine Co-Creativity with Older Adults -- A Learning Community to Study Explainable Dialogues (Marianne Bossema et al., 2023)

{{<citation>}}

Marianne Bossema, Rob Saunders, Somaya Ben Allouch. (2023)  
**Human-Machine Co-Creativity with Older Adults -- A Learning Community to Study Explainable Dialogues**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.07028v1)  

---


**ABSTRACT**  
This position paper is part of a long-term research project on human-machine co-creativity with older adults. The goal is to investigate how robots and AI-generated content can contribute to older adults' creative experiences, with a focus on collaborative drawing and painting. The research has recently started, and current activities are centred around literature studies, interviews with seniors and artists, and developing initial prototypes. In addition, a course "Drawing with Robots", is being developed to establish collaboration between human and machine learners: older adults, artists, students, researchers, and artificial agents. We present this course as a learning community and as an opportunity for studying how explainable AI and creative dialogues can be intertwined in human-machine co-creativity with older adults.

{{</citation>}}


### (56/94) Cleaning Up the Streets: Understanding Motivations, Mental Models, and Concerns of Users Flagging Social Media Posts (Alice Qian Zhang et al., 2023)

{{<citation>}}

Alice Qian Zhang, Kaitlin Montague, Shagun Jhaver. (2023)  
**Cleaning Up the Streets: Understanding Motivations, Mental Models, and Concerns of Users Flagging Social Media Posts**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2309.06688v1)  

---


**ABSTRACT**  
Social media platforms offer flagging, a technical feature that empowers users to report inappropriate posts or bad actors, to reduce online harms. While flags are often presented as flimsy icons, their simple interface disguises complex underlying interactions among users, algorithms, and moderators. Through semi-structured interviews with 22 active social media users who had recently flagged, we examine their understanding of flagging procedures, explore the factors that motivate and demotivate them from engaging in flagging, and surface their emotional, cognitive, and privacy concerns. Our findings show that a belief in generalized reciprocity motivates flag submissions, but deficiencies in procedural transparency create gaps in users' mental models of how platforms process flags. We highlight how flags raise questions about the distribution of labor and responsibility between platforms and users for addressing online harm. We recommend innovations in the flagging design space that assist user comprehension and facilitate granular status checks while aligning with their privacy and security expectations.

{{</citation>}}


### (57/94) Beyond English: Centering Multilingualism in Data Visualization (Noëlle Rakotondravony et al., 2023)

{{<citation>}}

Noëlle Rakotondravony, Priya Dhawka, Melanie Bancilhon. (2023)  
**Beyond English: Centering Multilingualism in Data Visualization**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.06659v1)  

---


**ABSTRACT**  
Information visualization and natural language are intricately linked. However, the majority of research and relevant work in information and data visualization (and human-computer interaction) involve English-speaking populations as both researchers and participants, are published in English, and are presented predominantly at English-speaking venues. Although several solutions can be proposed such as translating English texts in visualization to other languages, there is little research that looks at the intersection of data visualization and different languages, and the implications that current visualization practices have on non-English speaking communities. In this position paper, we argue that linguistically diverse communities abound beyond the English-speaking world and offer a richness of experiences for the visualization research community to engage with. Through a case study of how two non-English languages interplay with data visualization reasoning in Madagascar, we describe how monolingualism in data visualization impacts the experiences of underrepresented populations and emphasize potential harm to these communities. Lastly, we raise several questions towards advocating for more inclusive visualization practices that center the diverse experiences of linguistically underrepresented populations.

{{</citation>}}


## cs.LG (17)



### (58/94) Autotuning Apache TVM-based Scientific Applications Using Bayesian Optimization (Xingfu Wu et al., 2023)

{{<citation>}}

Xingfu Wu, Praveen Paramasivam, Valerie Taylor. (2023)  
**Autotuning Apache TVM-based Scientific Applications Using Bayesian Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NA, cs.LG, math-NA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.07235v1)  

---


**ABSTRACT**  
Apache TVM (Tensor Virtual Machine), an open source machine learning compiler framework designed to optimize computations across various hardware platforms, provides an opportunity to improve the performance of dense matrix factorizations such as LU (Lower Upper) decomposition and Cholesky decomposition on GPUs and AI (Artificial Intelligence) accelerators. In this paper, we propose a new TVM autotuning framework using Bayesian Optimization and use the TVM tensor expression language to implement linear algebra kernels such as LU, Cholesky, and 3mm. We use these scientific computation kernels to evaluate the effectiveness of our methods on a GPU cluster, called Swing, at Argonne National Laboratory. We compare the proposed autotuning framework with the TVM autotuning framework AutoTVM with four tuners and find that our framework outperforms AutoTVM in most cases.

{{</citation>}}


### (59/94) EarthPT: a foundation model for Earth Observation (Michael J. Smith et al., 2023)

{{<citation>}}

Michael J. Smith, Luke Fleming, James E. Geach. (2023)  
**EarthPT: a foundation model for Earth Observation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-geo-ph  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.07207v1)  

---


**ABSTRACT**  
We introduce EarthPT -- an Earth Observation (EO) pretrained transformer. EarthPT is a 700 million parameter decoding transformer foundation model trained in an autoregressive self-supervised manner and developed specifically with EO use-cases in mind. We demonstrate that EarthPT is an effective forecaster that can accurately predict future pixel-level surface reflectances across the 400-2300 nm range well into the future. For example, forecasts of the evolution of the Normalised Difference Vegetation Index (NDVI) have a typical error of approximately 0.05 (over a natural range of -1 -> 1) at the pixel level over a five month test set horizon, out-performing simple phase-folded models based on historical averaging. We also demonstrate that embeddings learnt by EarthPT hold semantically meaningful information and could be exploited for downstream tasks such as highly granular, dynamic land use classification. Excitingly, we note that the abundance of EO data provides us with -- in theory -- quadrillions of training tokens. Therefore, if we assume that EarthPT follows neural scaling laws akin to those derived for Large Language Models (LLMs), there is currently no data-imposed limit to scaling EarthPT and other similar `Large Observation Models.'

{{</citation>}}


### (60/94) PILOT: A Pre-Trained Model-Based Continual Learning Toolbox (Hai-Long Sun et al., 2023)

{{<citation>}}

Hai-Long Sun, Da-Wei Zhou, Han-Jia Ye, De-Chuan Zhan. (2023)  
**PILOT: A Pre-Trained Model-Based Continual Learning Toolbox**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2309.07117v1)  

---


**ABSTRACT**  
While traditional machine learning can effectively tackle a wide range of problems, it primarily operates within a closed-world setting, which presents limitations when dealing with streaming data. As a solution, incremental learning emerges to address real-world scenarios involving new data's arrival. Recently, pre-training has made significant advancements and garnered the attention of numerous researchers. The strong performance of these pre-trained models (PTMs) presents a promising avenue for developing continual learning algorithms that can effectively adapt to real-world scenarios. Consequently, exploring the utilization of PTMs in incremental learning has become essential. This paper introduces a pre-trained model-based continual learning toolbox known as PILOT. On the one hand, PILOT implements some state-of-the-art class-incremental learning algorithms based on pre-trained models, such as L2P, DualPrompt, and CODA-Prompt. On the other hand, PILOT also fits typical class-incremental learning algorithms (e.g., DER, FOSTER, and MEMO) within the context of pre-trained models to evaluate their effectiveness.

{{</citation>}}


### (61/94) Characterizing Speed Performance of Multi-Agent Reinforcement Learning (Samuel Wiggins et al., 2023)

{{<citation>}}

Samuel Wiggins, Yuan Meng, Rajgopal Kannan, Viktor Prasanna. (2023)  
**Characterizing Speed Performance of Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.07108v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning (MARL) has achieved significant success in large-scale AI systems and big-data applications such as smart grids, surveillance, etc. Existing advancements in MARL algorithms focus on improving the rewards obtained by introducing various mechanisms for inter-agent cooperation. However, these optimizations are usually compute- and memory-intensive, thus leading to suboptimal speed performance in end-to-end training time. In this work, we analyze the speed performance (i.e., latency-bounded throughput) as the key metric in MARL implementations. Specifically, we first introduce a taxonomy of MARL algorithms from an acceleration perspective categorized by (1) training scheme and (2) communication method. Using our taxonomy, we identify three state-of-the-art MARL algorithms - Multi-Agent Deep Deterministic Policy Gradient (MADDPG), Target-oriented Multi-agent Communication and Cooperation (ToM2C), and Networked Multi-Agent RL (NeurComm) - as target benchmark algorithms, and provide a systematic analysis of their performance bottlenecks on a homogeneous multi-core CPU platform. We justify the need for MARL latency-bounded throughput to be a key performance metric in future literature while also addressing opportunities for parallelization and acceleration.

{{</citation>}}


### (62/94) Mitigating Group Bias in Federated Learning for Heterogeneous Devices (Khotso Selialia et al., 2023)

{{<citation>}}

Khotso Selialia, Yasra Chandio, Fatima M. Anwar. (2023)  
**Mitigating Group Bias in Federated Learning for Heterogeneous Devices**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.07085v1)  

---


**ABSTRACT**  
Federated Learning is emerging as a privacy-preserving model training approach in distributed edge applications. As such, most edge deployments are heterogeneous in nature i.e., their sensing capabilities and environments vary across deployments. This edge heterogeneity violates the independence and identical distribution (IID) property of local data across clients and produces biased global models i.e. models that contribute to unfair decision-making and discrimination against a particular community or a group. Existing bias mitigation techniques only focus on bias generated from label heterogeneity in non-IID data without accounting for domain variations due to feature heterogeneity and do not address global group-fairness property.   Our work proposes a group-fair FL framework that minimizes group-bias while preserving privacy and without resource utilization overhead. Our main idea is to leverage average conditional probabilities to compute a cross-domain group \textit{importance weights} derived from heterogeneous training data to optimize the performance of the worst-performing group using a modified multiplicative weights update method. Additionally, we propose regularization techniques to minimize the difference between the worst and best-performing groups while making sure through our thresholding mechanism to strike a balance between bias reduction and group performance degradation. Our evaluation of human emotion recognition and image classification benchmarks assesses the fair decision-making of our framework in real-world heterogeneous settings.

{{</citation>}}


### (63/94) Unsupervised Contrast-Consistent Ranking with Language Models (Niklas Stoehr et al., 2023)

{{<citation>}}

Niklas Stoehr, Pengxiang Cheng, Jing Wang, Daniel Preotiuc-Pietro, Rajarshi Bhowmik. (2023)  
**Unsupervised Contrast-Consistent Ranking with Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.06991v1)  

---


**ABSTRACT**  
Language models contain ranking-based knowledge and are powerful solvers of in-context ranking tasks. For instance, they may have parametric knowledge about the ordering of countries by size or may be able to rank reviews by sentiment. Recent work focuses on pairwise, pointwise, and listwise prompting techniques to elicit a language model's ranking knowledge. However, we find that even with careful calibration and constrained decoding, prompting-based techniques may not always be self-consistent in the rankings they produce. This motivates us to explore an alternative approach that is inspired by an unsupervised probing method called Contrast-Consistent Search (CCS). The idea is to train a probing model guided by a logical constraint: a model's representation of a statement and its negation must be mapped to contrastive true-false poles consistently across multiple statements. We hypothesize that similar constraints apply to ranking tasks where all items are related via consistent pairwise or listwise comparisons. To this end, we extend the binary CCS method to Contrast-Consistent Ranking (CCR) by adapting existing ranking methods such as the Max-Margin Loss, Triplet Loss, and Ordinal Regression objective. Our results confirm that, for the same language model, CCR probing outperforms prompting and even performs on a par with prompting much larger language models.

{{</citation>}}


### (64/94) Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments (Simon Queyrut et al., 2023)

{{<citation>}}

Simon Queyrut, Valerio Schiavoni, Pascal Felber. (2023)  
**Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack, Attention, ImageNet, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.07197v1)  

---


**ABSTRACT**  
The main premise of federated learning (FL) is that machine learning model updates are computed locally to preserve user data privacy. This approach avoids by design user data to ever leave the perimeter of their device. Once the updates aggregated, the model is broadcast to all nodes in the federation. However, without proper defenses, compromised nodes can probe the model inside their local memory in search for adversarial examples, which can lead to dangerous real-world scenarios. For instance, in image-based applications, adversarial examples consist of images slightly perturbed to the human eye getting misclassified by the local model. These adversarial images are then later presented to a victim node's counterpart model to replay the attack. Typical examples harness dissemination strategies such as altered traffic signs (patch attacks) no longer recognized by autonomous vehicles or seemingly unaltered samples that poison the local dataset of the FL scheme to undermine its robustness. Pelta is a novel shielding mechanism leveraging Trusted Execution Environments (TEEs) that reduce the ability of attackers to craft adversarial samples. Pelta masks inside the TEE the first part of the back-propagation chain rule, typically exploited by attackers to craft the malicious samples. We evaluate Pelta on state-of-the-art accurate models using three well-established datasets: CIFAR-10, CIFAR-100 and ImageNet. We show the effectiveness of Pelta in mitigating six white-box state-of-the-art adversarial attacks, such as Projected Gradient Descent, Momentum Iterative Method, Auto Projected Gradient Descent, the Carlini & Wagner attack. In particular, Pelta constitutes the first attempt at defending an ensemble model against the Self-Attention Gradient attack to the best of our knowledge. Our code is available to the research community at https://github.com/queyrusi/Pelta.

{{</citation>}}


### (65/94) DNNShifter: An Efficient DNN Pruning System for Edge Computing (Bailey J. Eccles et al., 2023)

{{<citation>}}

Bailey J. Eccles, Philip Rodgers, Peter Kilpatrick, Ivor Spence, Blesson Varghese. (2023)  
**DNNShifter: An Efficient DNN Pruning System for Edge Computing**  

---
Primary Category: cs.LG  
Categories: 68T07, I-2-1, cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.06973v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) underpin many machine learning applications. Production quality DNN models achieve high inference accuracy by training millions of DNN parameters which has a significant resource footprint. This presents a challenge for resources operating at the extreme edge of the network, such as mobile and embedded devices that have limited computational and memory resources. To address this, models are pruned to create lightweight, more suitable variants for these devices. Existing pruning methods are unable to provide similar quality models compared to their unpruned counterparts without significant time costs and overheads or are limited to offline use cases. Our work rapidly derives suitable model variants while maintaining the accuracy of the original model. The model variants can be swapped quickly when system and network conditions change to match workload demand. This paper presents DNNShifter, an end-to-end DNN training, spatial pruning, and model switching system that addresses the challenges mentioned above. At the heart of DNNShifter is a novel methodology that prunes sparse models using structured pruning. The pruned model variants generated by DNNShifter are smaller in size and thus faster than dense and sparse model predecessors, making them suitable for inference at the edge while retaining near similar accuracy as of the original dense model. DNNShifter generates a portfolio of model variants that can be swiftly interchanged depending on operational conditions. DNNShifter produces pruned model variants up to 93x faster than conventional training methods. Compared to sparse models, the pruned model variants are up to 5.14x smaller and have a 1.67x inference latency speedup, with no compromise to sparse model accuracy. In addition, DNNShifter has up to 11.9x lower overhead for switching models and up to 3.8x lower memory utilisation than existing approaches.

{{</citation>}}


### (66/94) Attention-based Dynamic Graph Convolutional Recurrent Neural Network for Traffic Flow Prediction in Highway Transportation (Tianpu Zhang et al., 2023)

{{<citation>}}

Tianpu Zhang, Weilong Ding, Mengda Xing. (2023)  
**Attention-based Dynamic Graph Convolutional Recurrent Neural Network for Traffic Flow Prediction in Highway Transportation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-GR, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.07196v1)  

---


**ABSTRACT**  
As one of the important tools for spatial feature extraction, graph convolution has been applied in a wide range of fields such as traffic flow prediction. However, current popular works of graph convolution cannot guarantee spatio-temporal consistency in a long period. The ignorance of correlational dynamics, convolutional locality and temporal comprehensiveness would limit predictive accuracy. In this paper, a novel Attention-based Dynamic Graph Convolutional Recurrent Neural Network (ADGCRNN) is proposed to improve traffic flow prediction in highway transportation. Three temporal resolutions of data sequence are effectively integrated by self-attention to extract characteristics; multi-dynamic graphs and their weights are dynamically created to compliantly combine the varying characteristics; a dedicated gated kernel emphasizing highly relative nodes is introduced on these complete graphs to reduce overfitting for graph convolution operations. Experiments on two public datasets show our work better than state-of-the-art baselines, and case studies of a real Web system prove practical benefit in highway transportation.

{{</citation>}}


### (67/94) Domain-Aware Augmentations for Unsupervised Online General Continual Learning (Nicolas Michel et al., 2023)

{{<citation>}}

Nicolas Michel, Romain Negrel, Giovanni Chierchia, Jean-François Bercher. (2023)  
**Domain-Aware Augmentations for Unsupervised Online General Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.06896v1)  

---


**ABSTRACT**  
Continual Learning has been challenging, especially when dealing with unsupervised scenarios such as Unsupervised Online General Continual Learning (UOGCL), where the learning agent has no prior knowledge of class boundaries or task change information. While previous research has focused on reducing forgetting in supervised setups, recent studies have shown that self-supervised learners are more resilient to forgetting. This paper proposes a novel approach that enhances memory usage for contrastive learning in UOGCL by defining and using stream-dependent data augmentations together with some implementation tricks. Our proposed method is simple yet effective, achieves state-of-the-art results compared to other unsupervised approaches in all considered setups, and reduces the gap between supervised and unsupervised continual learning. Our domain-aware augmentation procedure can be adapted to other replay-based methods, making it a promising strategy for continual learning.

{{</citation>}}


### (68/94) Safe Reinforcement Learning with Dual Robustness (Zeyang Li et al., 2023)

{{<citation>}}

Zeyang Li, Chuxiong Hu, Yunan Wang, Yujie Yang, Shengbo Eben Li. (2023)  
**Safe Reinforcement Learning with Dual Robustness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06835v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) agents are vulnerable to adversarial disturbances, which can deteriorate task performance or compromise safety specifications. Existing methods either address safety requirements under the assumption of no adversary (e.g., safe RL) or only focus on robustness against performance adversaries (e.g., robust RL). Learning one policy that is both safe and robust remains a challenging open problem. The difficulty is how to tackle two intertwined aspects in the worst cases: feasibility and optimality. Optimality is only valid inside a feasible region, while identification of maximal feasible region must rely on learning the optimal policy. To address this issue, we propose a systematic framework to unify safe RL and robust RL, including problem formulation, iteration scheme, convergence analysis and practical algorithm design. This unification is built upon constrained two-player zero-sum Markov games. A dual policy iteration scheme is proposed, which simultaneously optimizes a task policy and a safety policy. The convergence of this iteration scheme is proved. Furthermore, we design a deep RL algorithm for practical implementation, called dually robust actor-critic (DRAC). The evaluations with safety-critical benchmarks demonstrate that DRAC achieves high performance and persistent safety under all scenarios (no adversary, safety adversary, performance adversary), outperforming all baselines significantly.

{{</citation>}}


### (69/94) FedDIP: Federated Learning with Extreme Dynamic Pruning and Incremental Regularization (Qianyu Long et al., 2023)

{{<citation>}}

Qianyu Long, Christos Anagnostopoulos, Shameem Puthiya Parambath, Daning Bi. (2023)  
**FedDIP: Federated Learning with Extreme Dynamic Pruning and Incremental Regularization**  

---
Primary Category: cs.LG  
Categories: H-4; I-2, cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.06805v1)  

---


**ABSTRACT**  
Federated Learning (FL) has been successfully adopted for distributed training and inference of large-scale Deep Neural Networks (DNNs). However, DNNs are characterized by an extremely large number of parameters, thus, yielding significant challenges in exchanging these parameters among distributed nodes and managing the memory. Although recent DNN compression methods (e.g., sparsification, pruning) tackle such challenges, they do not holistically consider an adaptively controlled reduction of parameter exchange while maintaining high accuracy levels. We, therefore, contribute with a novel FL framework (coined FedDIP), which combines (i) dynamic model pruning with error feedback to eliminate redundant information exchange, which contributes to significant performance improvement, with (ii) incremental regularization that can achieve \textit{extreme} sparsity of models. We provide convergence analysis of FedDIP and report on a comprehensive performance and comparative assessment against state-of-the-art methods using benchmark data sets and DNN models. Our results showcase that FedDIP not only controls the model sparsity but efficiently achieves similar or better performance compared to other model pruning methods adopting incremental regularization during distributed model training. The code is available at: https://github.com/EricLoong/feddip.

{{</citation>}}


### (70/94) Electricity Demand Forecasting through Natural Language Processing with Long Short-Term Memory Networks (Yun Bai et al., 2023)

{{<citation>}}

Yun Bai, Simon Camal, Andrea Michiorri. (2023)  
**Electricity Demand Forecasting through Natural Language Processing with Long Short-Term Memory Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.06793v1)  

---


**ABSTRACT**  
Electricity demand forecasting is a well established research field. Usually this task is performed considering historical loads, weather forecasts, calendar information and known major events. Recently attention has been given on the possible use of new sources of information from textual news in order to improve the performance of these predictions. This paper proposes a Long and Short-Term Memory (LSTM) network incorporating textual news features that successfully predicts the deterministic and probabilistic tasks of the UK national electricity demand. The study finds that public sentiment and word vector representations related to transport and geopolitics have time-continuity effects on electricity demand. The experimental results show that the LSTM with textual features improves by more than 3% compared to the pure LSTM benchmark and by close to 10% over the official benchmark. Furthermore, the proposed model effectively reduces forecasting uncertainty by narrowing the confidence interval and bringing the forecast distribution closer to the truth.

{{</citation>}}


### (71/94) MCNS: Mining Causal Natural Structures Inside Time Series via A Novel Internal Causality Scheme (Yuanhao Liu et al., 2023)

{{<citation>}}

Yuanhao Liu, Dehui Du, Zihan Jiang, Anyan Huang, Yiyang Li. (2023)  
**MCNS: Mining Causal Natural Structures Inside Time Series via A Novel Internal Causality Scheme**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.06739v1)  

---


**ABSTRACT**  
Causal inference permits us to discover covert relationships of various variables in time series. However, in most existing works, the variables mentioned above are the dimensions. The causality between dimensions could be cursory, which hinders the comprehension of the internal relationship and the benefit of the causal graph to the neural networks (NNs). In this paper, we find that causality exists not only outside but also inside the time series because it reflects a succession of events in the real world. It inspires us to seek the relationship between internal subsequences. However, the challenges are the hardship of discovering causality from subsequences and utilizing the causal natural structures to improve NNs. To address these challenges, we propose a novel framework called Mining Causal Natural Structure (MCNS), which is automatic and domain-agnostic and helps to find the causal natural structures inside time series via the internal causality scheme. We evaluate the MCNS framework and impregnation NN with MCNS on time series classification tasks. Experimental results illustrate that our impregnation, by refining attention, shape selection classification, and pruning datasets, drives NN, even the data itself preferable accuracy and interpretability. Besides, MCNS provides an in-depth, solid summary of the time series and datasets.

{{</citation>}}


### (72/94) Bias Amplification Enhances Minority Group Performance (Gaotang Li et al., 2023)

{{<citation>}}

Gaotang Li, Jiarui Liu, Wei Hu. (2023)  
**Bias Amplification Enhances Minority Group Performance**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.06717v1)  

---


**ABSTRACT**  
Neural networks produced by standard training are known to suffer from poor accuracy on rare subgroups despite achieving high accuracy on average, due to the correlations between certain spurious features and labels. Previous approaches based on worst-group loss minimization (e.g. Group-DRO) are effective in improving worse-group accuracy but require expensive group annotations for all the training samples. In this paper, we focus on the more challenging and realistic setting where group annotations are only available on a small validation set or are not available at all. We propose BAM, a novel two-stage training algorithm: in the first stage, the model is trained using a bias amplification scheme via introducing a learnable auxiliary variable for each training sample; in the second stage, we upweight the samples that the bias-amplified model misclassifies, and then continue training the same model on the reweighted dataset. Empirically, BAM achieves competitive performance compared with existing methods evaluated on spurious correlation benchmarks in computer vision and natural language processing. Moreover, we find a simple stopping criterion based on minimum class accuracy difference that can remove the need for group annotations, with little or no loss in worst-group accuracy. We perform extensive analyses and ablations to verify the effectiveness and robustness of our algorithm in varying class and group imbalance ratios.

{{</citation>}}


### (73/94) Attention Loss Adjusted Prioritized Experience Replay (Zhuoying Chen et al., 2023)

{{<citation>}}

Zhuoying Chen, Huiping Li, Rizhong Wang. (2023)  
**Attention Loss Adjusted Prioritized Experience Replay**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.06684v1)  

---


**ABSTRACT**  
Prioritized Experience Replay (PER) is a technical means of deep reinforcement learning by selecting experience samples with more knowledge quantity to improve the training rate of neural network. However, the non-uniform sampling used in PER inevitably shifts the state-action space distribution and brings the estimation error of Q-value function. In this paper, an Attention Loss Adjusted Prioritized (ALAP) Experience Replay algorithm is proposed, which integrates the improved Self-Attention network with Double-Sampling mechanism to fit the hyperparameter that can regulate the importance sampling weights to eliminate the estimation error caused by PER. In order to verify the effectiveness and generality of the algorithm, the ALAP is tested with value-function based, policy-gradient based and multi-agent reinforcement learning algorithms in OPENAI gym, and comparison studies verify the advantage and efficiency of the proposed training framework.

{{</citation>}}


### (74/94) ConR: Contrastive Regularizer for Deep Imbalanced Regression (Mahsa Keramati et al., 2023)

{{<citation>}}

Mahsa Keramati, Lili Meng, R. David Evans. (2023)  
**ConR: Contrastive Regularizer for Deep Imbalanced Regression**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06651v1)  

---


**ABSTRACT**  
Imbalanced distributions are ubiquitous in real-world data. They create constraints on Deep Neural Networks to represent the minority labels and avoid bias towards majority labels. The extensive body of imbalanced approaches address categorical label spaces but fail to effectively extend to regression problems where the label space is continuous. Conversely, local and global correlations among continuous labels provide valuable insights towards effectively modelling relationships in feature space. In this work, we propose ConR, a contrastive regularizer that models global and local label similarities in feature space and prevents the features of minority samples from being collapsed into their majority neighbours. Serving the similarities of the predictions as an indicator of feature similarities, ConR discerns the dissagreements between the label space and feature space and imposes a penalty on these disagreements. ConR minds the continuous nature of label space with two main strategies in a contrastive manner: incorrect proximities are penalized proportionate to the label similarities and the correct ones are encouraged to model local similarities. ConR consolidates essential considerations into a generic, easy-to-integrate, and efficient method that effectively addresses deep imbalanced regression. Moreover, ConR is orthogonal to existing approaches and smoothly extends to uni- and multi-dimensional label spaces. Our comprehensive experiments show that ConR significantly boosts the performance of all the state-of-the-art methods on three large-scale deep imbalanced regression benchmarks. Our code is publicly available in https://github.com/BorealisAI/ConR.

{{</citation>}}


## cs.SD (3)



### (75/94) Weakly-Supervised Multi-Task Learning for Audio-Visual Speaker Verification (Anith Selvakumar et al., 2023)

{{<citation>}}

Anith Selvakumar, Homa Fashandi. (2023)  
**Weakly-Supervised Multi-Task Learning for Audio-Visual Speaker Verification**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2309.07115v1)  

---


**ABSTRACT**  
In this paper, we present a methodology for achieving robust multimodal person representations optimized for open-set audio-visual speaker verification. Distance Metric Learning (DML) approaches have typically dominated this problem space, owing to strong performance on new and unseen classes. In our work, we explored multitask learning techniques to further boost performance of the DML approach and show that an auxiliary task with weak labels can increase the compactness of the learned speaker representation. We also extend the Generalized end-to-end loss (GE2E) to multimodal inputs and demonstrate that it can achieve competitive performance in an audio-visual space. Finally, we introduce a non-synchronous audio-visual sampling random strategy during training time that has shown to improve generalization. Our network achieves state of the art performance for speaker verification, reporting 0.244%, 0.252%, 0.441% Equal Error Rate (EER) on the three official trial lists of VoxCeleb1-O/E/H, which is to our knowledge, the best published results on VoxCeleb1-E and VoxCeleb1-H.

{{</citation>}}


### (76/94) DCTTS: Discrete Diffusion Model with Contrastive Learning for Text-to-speech Generation (Zhichao Wu et al., 2023)

{{<citation>}}

Zhichao Wu, Qiulin Li, Sixing Liu, Qun Yang. (2023)  
**DCTTS: Discrete Diffusion Model with Contrastive Learning for Text-to-speech Generation**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.06787v1)  

---


**ABSTRACT**  
In the Text-to-speech(TTS) task, the latent diffusion model has excellent fidelity and generalization, but its expensive resource consumption and slow inference speed have always been a challenging. This paper proposes Discrete Diffusion Model with Contrastive Learning for Text-to-Speech Generation(DCTTS). The following contributions are made by DCTTS: 1) The TTS diffusion model based on discrete space significantly lowers the computational consumption of the diffusion model and improves sampling speed; 2) The contrastive learning method based on discrete space is used to enhance the alignment connection between speech and text and improve sampling quality; and 3) It uses an efficient text encoder to simplify the model's parameters and increase computational efficiency. The experimental results demonstrate that the approach proposed in this paper has outstanding speech synthesis quality and sampling speed while significantly reducing the resource consumption of diffusion model. The synthesized samples are available at https://github.com/lawtherWu/DCTTS.

{{</citation>}}


### (77/94) Attention-based Encoder-Decoder End-to-End Neural Diarization with Embedding Enhancer (Zhengyang Chen et al., 2023)

{{<citation>}}

Zhengyang Chen, Bing Han, Shuai Wang, Yanmin Qian. (2023)  
**Attention-based Encoder-Decoder End-to-End Neural Diarization with Embedding Enhancer**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2309.06672v1)  

---


**ABSTRACT**  
Deep neural network-based systems have significantly improved the performance of speaker diarization tasks. However, end-to-end neural diarization (EEND) systems often struggle to generalize to scenarios with an unseen number of speakers, while target speaker voice activity detection (TS-VAD) systems tend to be overly complex. In this paper, we propose a simple attention-based encoder-decoder network for end-to-end neural diarization (AED-EEND). In our training process, we introduce a teacher-forcing strategy to address the speaker permutation problem, leading to faster model convergence. For evaluation, we propose an iterative decoding method that outputs diarization results for each speaker sequentially. Additionally, we propose an Enhancer module to enhance the frame-level speaker embeddings, enabling the model to handle scenarios with an unseen number of speakers. We also explore replacing the transformer encoder with a Conformer architecture, which better models local information. Furthermore, we discovered that commonly used simulation datasets for speaker diarization have a much higher overlap ratio compared to real data. We found that using simulated training data that is more consistent with real data can achieve an improvement in consistency. Extensive experimental validation demonstrates the effectiveness of our proposed methodologies. Our best system achieved a new state-of-the-art diarization error rate (DER) performance on all the CALLHOME (10.08%), DIHARD II (24.64%), and AMI (13.00%) evaluation benchmarks, when no oracle voice activity detection (VAD) is used. Beyond speaker diarization, our AED-EEND system also shows remarkable competitiveness as a speech type detection model.

{{</citation>}}


## stat.ML (1)



### (78/94) Data Augmentation via Subgroup Mixup for Improving Fairness (Madeline Navarro et al., 2023)

{{<citation>}}

Madeline Navarro, Camille Little, Genevera I. Allen, Santiago Segarra. (2023)  
**Data Augmentation via Subgroup Mixup for Improving Fairness**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.07110v1)  

---


**ABSTRACT**  
In this work, we propose data augmentation via pairwise mixup across subgroups to improve group fairness. Many real-world applications of machine learning systems exhibit biases across certain groups due to under-representation or training data that reflects societal biases. Inspired by the successes of mixup for improving classification performance, we develop a pairwise mixup scheme to augment training data and encourage fair and accurate decision boundaries for all subgroups. Data augmentation for group fairness allows us to add new samples of underrepresented groups to balance subpopulations. Furthermore, our method allows us to use the generalization ability of mixup to improve both fairness and accuracy. We compare our proposed mixup to existing data augmentation and bias mitigation approaches on both synthetic simulations and real-world benchmark fair classification data, demonstrating that we are able to achieve fair outcomes with robust if not improved accuracy.

{{</citation>}}


## eess.AS (1)



### (79/94) Can Whisper perform speech-based in-context learning (Siyin Wang et al., 2023)

{{<citation>}}

Siyin Wang, Chao-Han Huck Yang, Ji Wu, Chao Zhang. (2023)  
**Can Whisper perform speech-based in-context learning**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.07081v1)  

---


**ABSTRACT**  
This paper investigates the in-context learning abilities of the Whisper automatic speech recognition (ASR) models released by OpenAI. A novel speech-based in-context learning (SICL) approach is proposed for test-time adaptation, which can reduce the word error rates (WERs) with only a small number of labelled speech samples without gradient descent. Language-level adaptation experiments using Chinese dialects showed that when applying SICL to isolated word ASR, consistent and considerable relative WER reductions can be achieved using Whisper models of any size on two dialects, which is on average 32.3%. A k-nearest-neighbours-based in-context example selection technique can be applied to further improve the efficiency of SICL, which can increase the average relative WER reduction to 36.4%. The findings are verified using speaker adaptation or continuous speech recognition tasks, and both achieved considerable relative WER reductions. Detailed quantitative analyses are also provided to shed light on SICL's adaptability to phonological variances and dialect-specific lexical nuances.

{{</citation>}}


## cs.CR (7)



### (80/94) A Comprehensive Analysis of the Role of Artificial Intelligence and Machine Learning in Modern Digital Forensics and Incident Response (Dipo Dunsin et al., 2023)

{{<citation>}}

Dipo Dunsin, Mohamed C. Ghanem, Karim Ouazzane, Vassil Vassilev. (2023)  
**A Comprehensive Analysis of the Role of Artificial Intelligence and Machine Learning in Modern Digital Forensics and Incident Response**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-NI, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.07064v1)  

---


**ABSTRACT**  
In the dynamic landscape of digital forensics, the integration of Artificial Intelligence (AI) and Machine Learning (ML) stands as a transformative technology, poised to amplify the efficiency and precision of digital forensics investigations. However, the use of ML and AI in digital forensics is still in its nascent stages. As a result, this paper gives a thorough and in-depth analysis that goes beyond a simple survey and review. The goal is to look closely at how AI and ML techniques are used in digital forensics and incident response. This research explores cutting-edge research initiatives that cross domains such as data collection and recovery, the intricate reconstruction of cybercrime timelines, robust big data analysis, pattern recognition, safeguarding the chain of custody, and orchestrating responsive strategies to hacking incidents. This endeavour digs far beneath the surface to unearth the intricate ways AI-driven methodologies are shaping these crucial facets of digital forensics practice. While the promise of AI in digital forensics is evident, the challenges arising from increasing database sizes and evolving criminal tactics necessitate ongoing collaborative research and refinement within the digital forensics profession. This study examines the contributions, limitations, and gaps in the existing research, shedding light on the potential and limitations of AI and ML techniques. By exploring these different research areas, we highlight the critical need for strategic planning, continual research, and development to unlock AI's full potential in digital forensics and incident response. Ultimately, this paper underscores the significance of AI and ML integration in digital forensics, offering insights into their benefits, drawbacks, and broader implications for tackling modern cyber threats.

{{</citation>}}


### (81/94) Cryptography: Against AI and QAI Odds (Sheetal Harris et al., 2023)

{{<citation>}}

Sheetal Harris, Hassan Jalil Hadi, Umer Zukaib. (2023)  
**Cryptography: Against AI and QAI Odds**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2309.07022v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) presents prodigious technological prospects for development, however, all that glitters is not gold! The cyber-world faces the worst nightmare with the advent of AI and quantum computers. Together with Quantum Artificial Intelligence (QAI), they pose a catastrophic threat to modern cryptography. It would also increase the capability of cryptanalysts manifold, with its built-in persistent and extensive predictive intelligence. This prediction ability incapacitates the constrained message space in device cryptography. With the comparison of these assumptions and the intercepted ciphertext, the code-cracking process will considerably accelerate. Before the vigorous and robust developments in AI, we have never faced and never had to prepare for such a plaintext-originating attack. The supremacy of AI can be challenged by creating ciphertexts that would give the AI attacker erroneous responses stymied by randomness and misdirect them. AI threat is deterred by deviating from the conventional use of small, known-size keys and pattern-loaded ciphers. The strategy is vested in implementing larger secret size keys, supplemented by ad-hoc unilateral randomness of unbound limitations and a pattern-devoid technique. The very large key size can be handled with low processing and computational burden to achieve desired unicity distances. The strategy against AI odds is feasible by implementing non-algorithmic randomness, large and inexpensive memory chips, and wide-area communication networks. The strength of AI, i.e., randomness and pattern detection can be used to generate highly optimized ciphers and algorithms. These pattern-devoid, randomness-rich ciphers also provide a timely and plausible solution for NIST's proactive approach toward the quantum challenge.

{{</citation>}}


### (82/94) Communication-Efficient Laplace Mechanism for Differential Privacy via Random Quantization (Ali Moradi Shahmiri et al., 2023)

{{<citation>}}

Ali Moradi Shahmiri, Chih Wei Ling, Cheuk Ting Li. (2023)  
**Communication-Efficient Laplace Mechanism for Differential Privacy via Random Quantization**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.06982v1)  

---


**ABSTRACT**  
We propose the first method that realizes the Laplace mechanism exactly (i.e., a Laplace noise is added to the data) that requires only a finite amount of communication (whereas the original Laplace mechanism requires the transmission of a real number) while guaranteeing privacy against the server and database. Our mechanism can serve as a drop-in replacement for local or centralized differential privacy applications where the Laplace mechanism is used. Our mechanism is constructed using a random quantization technique. Unlike the simple and prevalent Laplace-mechanism-then-quantize approach, the quantization in our mechanism does not result in any distortion or degradation of utility. Unlike existing dithered quantization and channel simulation schemes for simulating additive Laplacian noise, our mechanism guarantees privacy not only against the database and downstream, but also against the honest but curious server which attempts to decode the data using the dither signals.

{{</citation>}}


### (83/94) MASTERKEY: Practical Backdoor Attack Against Speaker Verification Systems (Hanqing Guo et al., 2023)

{{<citation>}}

Hanqing Guo, Xun Chen, Junfeng Guo, Li Xiao, Qiben Yan. (2023)  
**MASTERKEY: Practical Backdoor Attack Against Speaker Verification Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs-SD, cs.CR, eess-AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2309.06981v1)  

---


**ABSTRACT**  
Speaker Verification (SV) is widely deployed in mobile systems to authenticate legitimate users by using their voice traits. In this work, we propose a backdoor attack MASTERKEY, to compromise the SV models. Different from previous attacks, we focus on a real-world practical setting where the attacker possesses no knowledge of the intended victim. To design MASTERKEY, we investigate the limitation of existing poisoning attacks against unseen targets. Then, we optimize a universal backdoor that is capable of attacking arbitrary targets. Next, we embed the speaker's characteristics and semantics information into the backdoor, making it imperceptible. Finally, we estimate the channel distortion and integrate it into the backdoor. We validate our attack on 6 popular SV models. Specifically, we poison a total of 53 models and use our trigger to attack 16,430 enrolled speakers, composed of 310 target speakers enrolled in 53 poisoned models. Our attack achieves 100% attack success rate with a 15% poison rate. By decreasing the poison rate to 3%, the attack success rate remains around 50%. We validate our attack in 3 real-world scenarios and successfully demonstrate the attack through both over-the-air and over-the-telephony-line scenarios.

{{</citation>}}


### (84/94) PhantomSound: Black-Box, Query-Efficient Audio Adversarial Attack via Split-Second Phoneme Injection (Hanqing Guo et al., 2023)

{{<citation>}}

Hanqing Guo, Guangjing Wang, Yuanda Wang, Bocheng Chen, Qiben Yan, Li Xiao. (2023)  
**PhantomSound: Black-Box, Query-Efficient Audio Adversarial Attack via Split-Second Phoneme Injection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-HC, cs.CR  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.06960v1)  

---


**ABSTRACT**  
In this paper, we propose PhantomSound, a query-efficient black-box attack toward voice assistants. Existing black-box adversarial attacks on voice assistants either apply substitution models or leverage the intermediate model output to estimate the gradients for crafting adversarial audio samples. However, these attack approaches require a significant amount of queries with a lengthy training stage. PhantomSound leverages the decision-based attack to produce effective adversarial audios, and reduces the number of queries by optimizing the gradient estimation. In the experiments, we perform our attack against 4 different speech-to-text APIs under 3 real-world scenarios to demonstrate the real-time attack impact. The results show that PhantomSound is practical and robust in attacking 5 popular commercial voice controllable devices over the air, and is able to bypass 3 liveness detection mechanisms with >95% success rate. The benchmark result shows that PhantomSound can generate adversarial examples and launch the attack in a few minutes. We significantly enhance the query efficiency and reduce the cost of a successful untargeted and targeted adversarial attack by 93.1% and 65.5% compared with the state-of-the-art black-box attacks, using merely ~300 queries (~5 minutes) and ~1,500 queries (~25 minutes), respectively.

{{</citation>}}


### (85/94) ZKROWNN: Zero Knowledge Right of Ownership for Neural Networks (Nojan Sheybani et al., 2023)

{{<citation>}}

Nojan Sheybani, Zahra Ghodsi, Ritvik Kapila, Farinaz Koushanfar. (2023)  
**ZKROWNN: Zero Knowledge Right of Ownership for Neural Networks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06779v1)  

---


**ABSTRACT**  
Training contemporary AI models requires investment in procuring learning data and computing resources, making the models intellectual property of the owners. Popular model watermarking solutions rely on key input triggers for detection; the keys have to be kept private to prevent discovery, forging, and removal of the hidden signatures. We present ZKROWNN, the first automated end-to-end framework utilizing Zero-Knowledge Proofs (ZKP) that enable an entity to validate their ownership of a model, while preserving the privacy of the watermarks. ZKROWNN permits a third party client to verify model ownership in less than a second, requiring as little as a few KBs of communication.

{{</citation>}}


### (86/94) DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass (Minxin Du et al., 2023)

{{<citation>}}

Minxin Du, Xiang Yue, Sherman S. M. Chow, Tianhao Wang, Chenyu Huang, Huan Sun. (2023)  
**DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.06746v1)  

---


**ABSTRACT**  
Differentially private stochastic gradient descent (DP-SGD) adds noise to gradients in back-propagation, safeguarding training data from privacy leakage, particularly membership inference. It fails to cover (inference-time) threats like embedding inversion and sensitive attribute inference. It is also costly in storage and computation when used to fine-tune large pre-trained language models (LMs).   We propose DP-Forward, which directly perturbs embedding matrices in the forward pass of LMs. It satisfies stringent local DP requirements for training and inference data. To instantiate it using the smallest matrix-valued noise, we devise an analytic matrix Gaussian~mechanism (aMGM) by drawing possibly non-i.i.d. noise from a matrix Gaussian distribution. We then investigate perturbing outputs from different hidden (sub-)layers of LMs with aMGM noises. Its utility on three typical tasks almost hits the non-private baseline and outperforms DP-SGD by up to 7.7pp at a moderate privacy level. It saves 3$\times$ time and memory costs compared to DP-SGD with the latest high-speed library. It also reduces the average success rates of embedding inversion and sensitive attribute inference by up to 88pp and 41pp, respectively, whereas DP-SGD fails.

{{</citation>}}


## cs.SE (1)



### (87/94) APICom: Automatic API Completion via Prompt Learning and Adversarial Training-based Data Augmentation (Yafeng Gu et al., 2023)

{{<citation>}}

Yafeng Gu, Yiheng Shen, Xiang Chen, Shaoyu Yang, Yiling Huang, Zhixiang Cao. (2023)  
**APICom: Automatic API Completion via Prompt Learning and Adversarial Training-based Data Augmentation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Adversarial Training, Augmentation  
[Paper Link](http://arxiv.org/abs/2309.07026v1)  

---


**ABSTRACT**  
Based on developer needs and usage scenarios, API (Application Programming Interface) recommendation is the process of assisting developers in finding the required API among numerous candidate APIs. Previous studies mainly modeled API recommendation as the recommendation task, which can recommend multiple candidate APIs for the given query, and developers may not yet be able to find what they need. Motivated by the neural machine translation research domain, we can model this problem as the generation task, which aims to directly generate the required API for the developer query. After our preliminary investigation, we find the performance of this intuitive approach is not promising. The reason is that there exists an error when generating the prefixes of the API. However, developers may know certain API prefix information during actual development in most cases. Therefore, we model this problem as the automatic completion task and propose a novel approach APICom based on prompt learning, which can generate API related to the query according to the prompts (i.e., API prefix information). Moreover, the effectiveness of APICom highly depends on the quality of the training dataset. In this study, we further design a novel gradient-based adversarial training method {\atpart} for data augmentation, which can improve the normalized stability when generating adversarial examples. To evaluate the effectiveness of APICom, we consider a corpus of 33k developer queries and corresponding APIs. Compared with the state-of-the-art baselines, our experimental results show that APICom can outperform all baselines by at least 40.02\%, 13.20\%, and 16.31\% in terms of the performance measures EM@1, MRR, and MAP. Finally, our ablation studies confirm the effectiveness of our component setting (such as our designed adversarial training method, our used pre-trained model, and prompt learning) in APICom.

{{</citation>}}


## cs.MA (1)



### (88/94) Enhancing the Performance of Multi-Agent Reinforcement Learning for Controlling HVAC Systems (Daniel Bayer et al., 2023)

{{<citation>}}

Daniel Bayer, Marco Pruckner. (2023)  
**Enhancing the Performance of Multi-Agent Reinforcement Learning for Controlling HVAC Systems**  

---
Primary Category: cs.MA  
Categories: I-2-1, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06940v1)  

---


**ABSTRACT**  
Systems for heating, ventilation and air-conditioning (HVAC) of buildings are traditionally controlled by a rule-based approach. In order to reduce the energy consumption and the environmental impact of HVAC systems more advanced control methods such as reinforcement learning are promising. Reinforcement learning (RL) strategies offer a good alternative, as user feedback can be integrated more easily and presence can also be incorporated. Moreover, multi-agent RL approaches scale well and can be generalized. In this paper, we propose a multi-agent RL framework based on existing work that learns reducing on one hand energy consumption by optimizing HVAC control and on the other hand user feedback by occupants about uncomfortable room temperatures. Second, we show how to reduce training time required for proper RL-agent-training by using parameter sharing between the multiple agents and apply different pretraining techniques. Results show that our framework is capable of reducing the energy by around 6% when controlling a complete building or 8% for a single room zone. The occupants complaints are acceptable or even better compared to a rule-based baseline. Additionally, our performance analysis show that the training time can be drastically reduced by using parameter sharing.

{{</citation>}}


## cs.AI (3)



### (89/94) Collectionless Artificial Intelligence (Marco Gori et al., 2023)

{{<citation>}}

Marco Gori, Stefano Melacci. (2023)  
**Collectionless Artificial Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06938v1)  

---


**ABSTRACT**  
By and large, the professional handling of huge data collections is regarded as a fundamental ingredient of the progress of machine learning and of its spectacular results in related disciplines, with a growing agreement on risks connected to the centralization of such data collections. This paper sustains the position that the time has come for thinking of new learning protocols where machines conquer cognitive skills in a truly human-like context centered on environmental interactions. This comes with specific restrictions on the learning protocol according to the collectionless principle, which states that, at each time instant, data acquired from the environment is processed with the purpose of contributing to update the current internal representation of the environment, and that the agent is not given the privilege of recording the temporal stream. Basically, there is neither permission to store the temporal information coming from the sensors, thus promoting the development of self-organized memorization skills at a more abstract level, instead of relying on bare storage to simulate learning dynamics that are typical of offline learning algorithms. This purposely extreme position is intended to stimulate the development of machines that learn to dynamically organize the information by following human-based schemes. The proposition of this challenge suggests developing new foundations on computational processes of learning and reasoning that might open the doors to a truly orthogonal competitive track on AI technologies that avoid data accumulation by design, thus offering a framework which is better suited concerning privacy issues, control and customizability. Finally, pushing towards massively distributed computation, the collectionless approach to AI will likely reduce the concentration of power in companies and governments, thus better facing geopolitical issues.

{{</citation>}}


### (90/94) When Geoscience Meets Foundation Models: Towards General Geoscience Artificial Intelligence System (Hao Zhang et al., 2023)

{{<citation>}}

Hao Zhang, Jin-Jian Xu. (2023)  
**When Geoscience Meets Foundation Models: Towards General Geoscience Artificial Intelligence System**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, physics-geo-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06799v1)  

---


**ABSTRACT**  
Geoscience foundation models represent a revolutionary approach in the field of Earth sciences by integrating massive cross-disciplinary data to simulate and understand the Earth systems dynamics. As a data-centric artificial intelligence (AI) paradigm, they uncover insights from petabytes of structured and unstructured data. Flexible task specification, diverse inputs and outputs and multi-modal knowledge representation enable comprehensive analysis infeasible with individual data sources. Critically, the scalability and generalizability of geoscience models allow for tackling diverse prediction, simulation, and decision challenges related to Earth systems interactions. Collaboration between domain experts and computer scientists leads to innovations in these invaluable tools for understanding the past, present, and future of our planet. However, challenges remain in validation and verification, scale, interpretability, knowledge representation, and social bias. Going forward, enhancing model integration, resolution, accuracy, and equity through cross-disciplinary teamwork is key. Despite current limitations, geoscience foundation models show promise for providing critical insights into pressing issues including climate change, natural hazards, and sustainability through their ability to probe scenarios and quantify uncertainties. Their continued evolution toward integrated, data-driven modeling holds paradigm-shifting potential for Earth science.

{{</citation>}}


### (91/94) TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models (Siyao Zhang et al., 2023)

{{<citation>}}

Siyao Zhang, Daocheng Fu, Zhao Zhang, Bin Yu, Pinlong Cai. (2023)  
**TrafficGPT: Viewing, Processing and Interacting with Traffic Foundation Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.06719v1)  

---


**ABSTRACT**  
With the promotion of chatgpt to the public, Large language models indeed showcase remarkable common sense, reasoning, and planning skills, frequently providing insightful guidance. These capabilities hold significant promise for their application in urban traffic management and control. However, LLMs struggle with addressing traffic issues, especially processing numerical data and interacting with simulations, limiting their potential in solving traffic-related challenges. In parallel, specialized traffic foundation models exist but are typically designed for specific tasks with limited input-output interactions. Combining these models with LLMs presents an opportunity to enhance their capacity for tackling complex traffic-related problems and providing insightful suggestions. To bridge this gap, we present TrafficGPT, a fusion of ChatGPT and traffic foundation models. This integration yields the following key enhancements: 1) empowering ChatGPT with the capacity to view, analyze, process traffic data, and provide insightful decision support for urban transportation system management; 2) facilitating the intelligent deconstruction of broad and complex tasks and sequential utilization of traffic foundation models for their gradual completion; 3) aiding human decision-making in traffic control through natural language dialogues; and 4) enabling interactive feedback and solicitation of revised outcomes. By seamlessly intertwining large language model and traffic expertise, TrafficGPT not only advances traffic management but also offers a novel approach to leveraging AI capabilities in this domain. The TrafficGPT demo can be found in https://github.com/lijlansg/TrafficGPT.git.

{{</citation>}}


## cs.IR (1)



### (92/94) Multi-behavior Recommendation with SVD Graph Neural Networks (Shengxi Fu et al., 2023)

{{<citation>}}

Shengxi Fu, Qianqian Ren. (2023)  
**Multi-behavior Recommendation with SVD Graph Neural Networks**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.06912v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) has been extensively employed in the field of recommender systems, offering users personalized recommendations and yielding remarkable outcomes. Recently, GNNs incorporating contrastive learning have demonstrated promising performance in handling sparse data problem of recommendation system. However, existing contrastive learning methods still have limitations in addressing the cold-start problem and resisting noise interference especially for multi-behavior recommendation. To mitigate the aforementioned issues, the present research posits a GNNs based multi-behavior recommendation model MB-SVD that utilizes Singular Value Decomposition (SVD) graphs to enhance model performance. In particular, MB-SVD considers user preferences under different behaviors, improving recommendation effectiveness while better addressing the cold-start problem. Our model introduces an innovative methodology, which subsume multi-behavior contrastive learning paradigm to proficiently discern the intricate interconnections among heterogeneous manifestations of user behavior and generates SVD graphs to automate the distillation of crucial multi-behavior self-supervised information for robust graph augmentation. Furthermore, the SVD based framework reduces the embedding dimensions and computational load. Thorough experimentation showcases the remarkable performance of our proposed MB-SVD approach in multi-behavior recommendation endeavors across diverse real-world datasets.

{{</citation>}}


## physics.data-an (1)



### (93/94) Scalable neural network models and terascale datasets for particle-flow reconstruction (Joosep Pata et al., 2023)

{{<citation>}}

Joosep Pata, Eric Wulff, Farouk Mokhtar, David Southwick, Mengke Zhang, Maria Girone, Javier Duarte. (2023)  
**Scalable neural network models and terascale datasets for particle-flow reconstruction**  

---
Primary Category: physics.data-an  
Categories: cs-LG, hep-ex, physics-data-an, physics-ins-det, physics.data-an, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06782v1)  

---


**ABSTRACT**  
We study scalable machine learning models for full event reconstruction in high-energy electron-positron collisions based on a highly granular detector simulation. Particle-flow (PF) reconstruction can be formulated as a supervised learning task using tracks and calorimeter clusters or hits. We compare a graph neural network and kernel-based transformer and demonstrate that both avoid quadratic memory allocation and computational cost while achieving realistic PF reconstruction. We show that hyperparameter tuning on a supercomputer significantly improves the physics performance of the models. We also demonstrate that the resulting model is highly portable across hardware processors, supporting Nvidia, AMD, and Intel Habana cards. Finally, we demonstrate that the model can be trained on highly granular inputs consisting of tracks and calorimeter hits, resulting in a competitive physics performance with the baseline. Datasets and software to reproduce the studies are published following the findable, accessible, interoperable, and reusable (FAIR) principles.

{{</citation>}}


## eess.SY (1)



### (94/94) A Multi-task Learning Framework for Drone State Identification and Trajectory Prediction (Antreas Palamas et al., 2023)

{{<citation>}}

Antreas Palamas, Nicolas Souli, Tania Panayiotou, Panayiotis Kolios, Georgios Ellinas. (2023)  
**A Multi-task Learning Framework for Drone State Identification and Trajectory Prediction**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.06741v1)  

---


**ABSTRACT**  
The rise of unmanned aerial vehicle (UAV) operations, as well as the vulnerability of the UAVs' sensors, has led to the need for proper monitoring systems for detecting any abnormal behavior of the UAV. This work addresses this problem by proposing an innovative multi-task learning framework (MLF-ST) for UAV state identification and trajectory prediction, that aims to optimize the performance of both tasks simultaneously. A deep neural network with shared layers to extract features from the input data is employed, utilizing drone sensor measurements and historical trajectory information. Moreover, a novel loss function is proposed that combines the two objectives, encouraging the network to jointly learn the features that are most useful for both tasks. The proposed MLF-ST framework is evaluated on a large dataset of UAV flights, illustrating that it is able to outperform various state-of-the-art baseline techniques in terms of both state identification and trajectory prediction. The evaluation of the proposed framework, using real-world data, demonstrates that it can enable applications such as UAV-based surveillance and monitoring, while also improving the safety and efficiency of UAV operations.

{{</citation>}}
