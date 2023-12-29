---
draft: false
title: "arXiv @ 2023.12.25"
date: 2023-12-25
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.25"
    identifier: arxiv_20231225
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (9)](#cscv-9)
- [cs.RO (3)](#csro-3)
- [cs.LG (13)](#cslg-13)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.CL (8)](#cscl-8)
- [cs.AI (8)](#csai-8)
- [cs.IR (2)](#csir-2)
- [cs.CR (3)](#cscr-3)
- [cs.MM (1)](#csmm-1)
- [cs.CE (1)](#csce-1)
- [cs.SE (3)](#csse-3)
- [cs.SD (2)](#cssd-2)
- [cs.DC (1)](#csdc-1)
- [eess.IV (1)](#eessiv-1)
- [cs.SI (1)](#cssi-1)

## cs.CV (9)



### (1/57) On the Promises and Challenges of Multimodal Foundation Models for Geographical, Environmental, Agricultural, and Urban Planning Applications (Chenjiao Tan et al., 2023)

{{<citation>}}

Chenjiao Tan, Qian Cao, Yiwei Li, Jielu Zhang, Xiao Yang, Huaqin Zhao, Zihao Wu, Zhengliang Liu, Hao Yang, Nemin Wu, Tao Tang, Xinyue Ye, Lilong Chai, Ninghao Liu, Changying Li, Lan Mu, Tianming Liu, Gengchen Mai. (2023)  
**On the Promises and Challenges of Multimodal Foundation Models for Geographical, Environmental, Agricultural, and Urban Planning Applications**  

---
Primary Category: cs.CV  
Categories: I-2-7; I-2-10; I-4-6; I-4-8; J-2, cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.17016v1)  

---


**ABSTRACT**  
The advent of large language models (LLMs) has heightened interest in their potential for multimodal applications that integrate language and vision. This paper explores the capabilities of GPT-4V in the realms of geography, environmental science, agriculture, and urban planning by evaluating its performance across a variety of tasks. Data sources comprise satellite imagery, aerial photos, ground-level images, field images, and public datasets. The model is evaluated on a series of tasks including geo-localization, textual data extraction from maps, remote sensing image classification, visual question answering, crop type identification, disease/pest/weed recognition, chicken behavior analysis, agricultural object counting, urban planning knowledge question answering, and plan generation. The results indicate the potential of GPT-4V in geo-localization, land cover classification, visual question answering, and basic image understanding. However, there are limitations in several tasks requiring fine-grained recognition and precise counting. While zero-shot learning shows promise, performance varies across problem domains and image complexities. The work provides novel insights into GPT-4V's capabilities and limitations for real-world geospatial, environmental, agricultural, and urban planning challenges. Further research should focus on augmenting the model's knowledge and reasoning for specialized domains through expanded training. Overall, the analysis demonstrates foundational multimodal intelligence, highlighting the potential of multimodal foundation models (FMs) to advance interdisciplinary applications at the nexus of computer vision and language.

{{</citation>}}


### (2/57) Towards Generalization in Subitizing with Neuro-Symbolic Loss using Holographic Reduced Representations (Mohammad Mahmudul Alam et al., 2023)

{{<citation>}}

Mohammad Mahmudul Alam, Edward Raff, Tim Oates. (2023)  
**Towards Generalization in Subitizing with Neuro-Symbolic Loss using Holographic Reduced Representations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, q-bio-NC  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.15310v1)  

---


**ABSTRACT**  
While deep learning has enjoyed significant success in computer vision tasks over the past decade, many shortcomings still exist from a Cognitive Science (CogSci) perspective. In particular, the ability to subitize, i.e., quickly and accurately identify the small (less than 6) count of items, is not well learned by current Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs) when using a standard cross-entropy (CE) loss. In this paper, we demonstrate that adapting tools used in CogSci research can improve the subitizing generalization of CNNs and ViTs by developing an alternative loss function using Holographic Reduced Representations (HRRs). We investigate how this neuro-symbolic approach to learning affects the subitizing capability of CNNs and ViTs, and so we focus on specially crafted problems that isolate generalization to specific aspects of subitizing. Via saliency maps and out-of-distribution performance, we are able to empirically observe that the proposed HRR loss improves subitizing generalization though it does not completely solve the problem. In addition, we find that ViTs perform considerably worse compared to CNNs in most respects on subitizing, except on one axis where an HRR-based loss provides improvement.

{{</citation>}}


### (3/57) Mitigating Algorithmic Bias on Facial Expression Recognition (Glauco Amigo et al., 2023)

{{<citation>}}

Glauco Amigo, Pablo Rivas Perea, Robert J. Marks. (2023)  
**Mitigating Algorithmic Bias on Facial Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.15307v1)  

---


**ABSTRACT**  
Biased datasets are ubiquitous and present a challenge for machine learning. For a number of categories on a dataset that are equally important but some are sparse and others are common, the learning algorithms will favor the ones with more presence. The problem of biased datasets is especially sensitive when dealing with minority people groups. How can we, from biased data, generate algorithms that treat every person equally? This work explores one way to mitigate bias using a debiasing variational autoencoder with experiments on facial expression recognition.

{{</citation>}}


### (4/57) Q-Boost: On Visual Quality Assessment Ability of Low-level Multi-Modality Foundation Models (Zicheng Zhang et al., 2023)

{{<citation>}}

Zicheng Zhang, Haoning Wu, Zhongpeng Ji, Chunyi Li, Erli Zhang, Wei Sun, Xiaohong Liu, Xiongkuo Min, Fengyu Sun, Shangling Jui, Weisi Lin, Guangtao Zhai. (2023)  
**Q-Boost: On Visual Quality Assessment Ability of Low-level Multi-Modality Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.15300v1)  

---


**ABSTRACT**  
Recent advancements in Multi-modality Large Language Models (MLLMs) have demonstrated remarkable capabilities in complex high-level vision tasks. However, the exploration of MLLM potential in visual quality assessment, a vital aspect of low-level vision, remains limited. To address this gap, we introduce Q-Boost, a novel strategy designed to enhance low-level MLLMs in image quality assessment (IQA) and video quality assessment (VQA) tasks, which is structured around two pivotal components: 1) Triadic-Tone Integration: Ordinary prompt design simply oscillates between the binary extremes of $positive$ and $negative$. Q-Boost innovates by incorporating a `middle ground' approach through $neutral$ prompts, allowing for a more balanced and detailed assessment. 2) Multi-Prompt Ensemble: Multiple quality-centric prompts are used to mitigate bias and acquire more accurate evaluation. The experimental results show that the low-level MLLMs exhibit outstanding zeros-shot performance on the IQA/VQA tasks equipped with the Q-Boost strategy.

{{</citation>}}


### (5/57) Wavelet Packet Power Spectrum Kullback-Leibler Divergence: A New Metric for Image Synthesis (Lokesh Veeramacheneni et al., 2023)

{{<citation>}}

Lokesh Veeramacheneni, Moritz Wolter, Juergen Gall. (2023)  
**Wavelet Packet Power Spectrum Kullback-Leibler Divergence: A New Metric for Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.15289v1)  

---


**ABSTRACT**  
Current metrics for generative neural networks are biased towards low frequencies, specific generators, objects from the ImageNet dataset, and value texture more than shape. Many current quality metrics do not measure frequency information directly. In response, we propose a new frequency band-based quality metric, which opens a door into the frequency domain yet, at the same time, preserves spatial aspects of the data. Our metric works well even if the distributions we compare are far from ImageNet or have been produced by differing generator architectures. We verify the quality of our metric by sampling a broad selection of generative networks on a wide variety of data sets. A user study ensures our metric aligns with human perception. Furthermore, we show that frequency band guidance can improve the frequency domain fidelity of a current generative network.

{{</citation>}}


### (6/57) MGDepth: Motion-Guided Cost Volume For Self-Supervised Monocular Depth In Dynamic Scenarios (Kaichen Zhou et al., 2023)

{{<citation>}}

Kaichen Zhou, Jia-Xing Zhong, Jia-Wang Bian, Qian Xie, Jian-Qing Zheng, Niki Trigoni, Andrew Markham. (2023)  
**MGDepth: Motion-Guided Cost Volume For Self-Supervised Monocular Depth In Dynamic Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.15268v1)  

---


**ABSTRACT**  
Despite advancements in self-supervised monocular depth estimation, challenges persist in dynamic scenarios due to the dependence on assumptions about a static world. In this paper, we present MGDepth, a Motion-Guided Cost Volume Depth Net, to achieve precise depth estimation for both dynamic objects and static backgrounds, all while maintaining computational efficiency. To tackle the challenges posed by dynamic content, we incorporate optical flow and coarse monocular depth to create a novel static reference frame. This frame is then utilized to build a motion-guided cost volume in collaboration with the target frame. Additionally, to enhance the accuracy and resilience of the network structure, we introduce an attention-based depth net architecture to effectively integrate information from feature maps with varying resolutions. Compared to methods with similar computational costs, MGDepth achieves a significant reduction of approximately seven percent in root-mean-square error for self-supervised monocular depth estimation on the KITTI-2015 dataset.

{{</citation>}}


### (7/57) Self-Supervised Depth Completion Guided by 3D Perception and Geometry Consistency (Yu Cai et al., 2023)

{{<citation>}}

Yu Cai, Tianyu Shen, Shi-Sheng Huang, Hua Huang. (2023)  
**Self-Supervised Depth Completion Guided by 3D Perception and Geometry Consistency**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.15263v1)  

---


**ABSTRACT**  
Depth completion, aiming to predict dense depth maps from sparse depth measurements, plays a crucial role in many computer vision related applications. Deep learning approaches have demonstrated overwhelming success in this task. However, high-precision depth completion without relying on the ground-truth data, which are usually costly, still remains challenging. The reason lies on the ignorance of 3D structural information in most previous unsupervised solutions, causing inaccurate spatial propagation and mixed-depth problems. To alleviate the above challenges, this paper explores the utilization of 3D perceptual features and multi-view geometry consistency to devise a high-precision self-supervised depth completion method. Firstly, a 3D perceptual spatial propagation algorithm is constructed with a point cloud representation and an attention weighting mechanism to capture more reasonable and favorable neighboring features during the iterative depth propagation process. Secondly, the multi-view geometric constraints between adjacent views are explicitly incorporated to guide the optimization of the whole depth completion model in a self-supervised manner. Extensive experiments on benchmark datasets of NYU-Depthv2 and VOID demonstrate that the proposed model achieves the state-of-the-art depth completion performance compared with other unsupervised methods, and competitive performance compared with previous supervised methods.

{{</citation>}}


### (8/57) Scale Optimization Using Evolutionary Reinforcement Learning for Object Detection on Drone Imagery (Jialu Zhang et al., 2023)

{{<citation>}}

Jialu Zhang, Xiaoying Yang, Wentao He, Jianfeng Ren, Qian Zhang, Titian Zhao, Ruibin Bai, Xiangjian He, Jiang Liu. (2023)  
**Scale Optimization Using Evolutionary Reinforcement Learning for Object Detection on Drone Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone, Object Detection, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15219v1)  

---


**ABSTRACT**  
Object detection in aerial imagery presents a significant challenge due to large scale variations among objects. This paper proposes an evolutionary reinforcement learning agent, integrated within a coarse-to-fine object detection framework, to optimize the scale for more effective detection of objects in such images. Specifically, a set of patches potentially containing objects are first generated. A set of rewards measuring the localization accuracy, the accuracy of predicted labels, and the scale consistency among nearby patches are designed in the agent to guide the scale optimization. The proposed scale-consistency reward ensures similar scales for neighboring objects of the same category. Furthermore, a spatial-semantic attention mechanism is designed to exploit the spatial semantic relations between patches. The agent employs the proximal policy optimization strategy in conjunction with the evolutionary strategy, effectively utilizing both the current patch status and historical experience embedded in the agent. The proposed model is compared with state-of-the-art methods on two benchmark datasets for object detection on drone imagery. It significantly outperforms all the compared methods.

{{</citation>}}


### (9/57) Spatial-Temporal Decoupling Contrastive Learning for Skeleton-based Human Action Recognition (Shaojie Zhang et al., 2023)

{{<citation>}}

Shaojie Zhang, Jianqin Yin, Yonghao Dang. (2023)  
**Spatial-Temporal Decoupling Contrastive Learning for Skeleton-based Human Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.15144v1)  

---


**ABSTRACT**  
Skeleton-based action recognition is a central task of human-computer interaction. Current methods apply the modeling paradigm of image recognition to it directly. However, the skeleton sequences abstracted from the human body is a sparse representation. The features extracted from the skeleton encoder are spatiotemporal decoupled, which may confuse the semantics. To reduce the coupling and improve the semantics of the global features, we propose a framework (STD-CL) for skeleton-based action recognition. We first decouple the spatial-specific and temporal-specific features from the spatiotemporal features. Then we apply the attentive features to contrastive learning, which pulls together the features from the positive pairs and pushes away the feature embedding from the negative pairs. Moreover, the proposed training strategy STD-CL can be incorporated into current methods. Without additional compute consumption in the testing phase, our STD-CL with four various backbones (HCN, 2S-AGCN, CTR-GCN, and Hyperformer) achieves improvement on NTU60, NTU120, and NW-UCLA benchmarks. We will release our code at: https://github.com/BUPTSJZhang/STD-CL.

{{</citation>}}


## cs.RO (3)



### (10/57) WildScenes: A Benchmark for 2D and 3D Semantic Segmentation in Large-scale Natural Environments (Kavisha Vidanapathirana et al., 2023)

{{<citation>}}

Kavisha Vidanapathirana, Joshua Knights, Stephen Hausler, Mark Cox, Milad Ramezani, Jason Jooste, Ethan Griffiths, Shaheer Mohamed, Sridha Sridharan, Clinton Fookes, Peyman Moghadam. (2023)  
**WildScenes: A Benchmark for 2D and 3D Semantic Segmentation in Large-scale Natural Environments**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.15364v1)  

---


**ABSTRACT**  
Recent progress in semantic scene understanding has primarily been enabled by the availability of semantically annotated bi-modal (camera and lidar) datasets in urban environments. However, such annotated datasets are also needed for natural, unstructured environments to enable semantic perception for applications, including conservation, search and rescue, environment monitoring, and agricultural automation. Therefore, we introduce WildScenes, a bi-modal benchmark dataset consisting of multiple large-scale traversals in natural environments, including semantic annotations in high-resolution 2D images and dense 3D lidar point clouds, and accurate 6-DoF pose information. The data is (1) trajectory-centric with accurate localization and globally aligned point clouds, (2) calibrated and synchronized to support bi-modal inference, and (3) containing different natural environments over 6 months to support research on domain adaptation. Our 3D semantic labels are obtained via an efficient automated process that transfers the human-annotated 2D labels from multiple views into 3D point clouds, thus circumventing the need for expensive and time-consuming human annotation in 3D. We introduce benchmarks on 2D and 3D semantic segmentation and evaluate a variety of recent deep-learning techniques to demonstrate the challenges in semantic segmentation in natural environments. We propose train-val-test splits for standard benchmarks as well as domain adaptation benchmarks and utilize an automated split generation technique to ensure the balance of class label distributions. The data, evaluation scripts and pretrained models will be released upon acceptance at https://csiro-robotics.github.io/WildScenes.

{{</citation>}}


### (11/57) RoboFiSense: Attention-Based Robotic Arm Activity Recognition with WiFi Sensing (Rojin Zandi et al., 2023)

{{<citation>}}

Rojin Zandi, Kian Behzad, Elaheh Motamedi, Hojjat Salehinejad, Milad Siami. (2023)  
**RoboFiSense: Attention-Based Robotic Arm Activity Recognition with WiFi Sensing**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO, eess-SP  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.15345v2)  

---


**ABSTRACT**  
Despite the current surge of interest in autonomous robotic systems, robot activity recognition within restricted indoor environments remains a formidable challenge. Conventional methods for detecting and recognizing robotic arms' activities often rely on vision-based or light detection and ranging (LiDAR) sensors, which require line-of-sight (LoS) access and may raise privacy concerns, for example, in nursing facilities. This research pioneers an innovative approach harnessing channel state information (CSI) measured from WiFi signals, subtly influenced by the activity of robotic arms. We developed an attention-based network to classify eight distinct activities performed by a Franka Emika robotic arm in different situations. Our proposed bidirectional vision transformer-concatenated (BiVTC) methodology aspires to predict robotic arm activities accurately, even when trained on activities with different velocities, all without dependency on external or internal sensors or visual aids. Considering the high dependency of CSI data to the environment, motivated us to study the problem of sniffer location selection, by systematically changing the sniffer's location and collecting different sets of data. Finally, this paper also marks the first publication of the CSI data of eight distinct robotic arm activities, collectively referred to as RoboFiSense. This initiative aims to provide a benchmark dataset and baselines to the research community, fostering advancements in the field of robotics sensing.

{{</citation>}}


### (12/57) MARS: Multi-Scale Adaptive Robotics Vision for Underwater Object Detection and Domain Generalization (Lyes Saad Saoud et al., 2023)

{{<citation>}}

Lyes Saad Saoud, Lakmal Seneviratne, Irfan Hussain. (2023)  
**MARS: Multi-Scale Adaptive Robotics Vision for Underwater Object Detection and Domain Generalization**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.15275v1)  

---


**ABSTRACT**  
Underwater robotic vision encounters significant challenges, necessitating advanced solutions to enhance performance and adaptability. This paper presents MARS (Multi-Scale Adaptive Robotics Vision), a novel approach to underwater object detection tailored for diverse underwater scenarios. MARS integrates Residual Attention YOLOv3 with Domain-Adaptive Multi-Scale Attention (DAMSA) to enhance detection accuracy and adapt to different domains. During training, DAMSA introduces domain class-based attention, enabling the model to emphasize domain-specific features. Our comprehensive evaluation across various underwater datasets demonstrates MARS's performance. On the original dataset, MARS achieves a mean Average Precision (mAP) of 58.57\%, showcasing its proficiency in detecting critical underwater objects like echinus, starfish, holothurian, scallop, and waterweeds. This capability holds promise for applications in marine robotics, marine biology research, and environmental monitoring. Furthermore, MARS excels at mitigating domain shifts. On the augmented dataset, which incorporates all enhancements (+Domain +Residual+Channel Attention+Multi-Scale Attention), MARS achieves an mAP of 36.16\%. This result underscores its robustness and adaptability in recognizing objects and performing well across a range of underwater conditions. The source code for MARS is publicly available on GitHub at https://github.com/LyesSaadSaoud/MARS-Object-Detection/

{{</citation>}}


## cs.LG (13)



### (13/57) MaDi: Learning to Mask Distractions for Generalization in Visual Deep Reinforcement Learning (Bram Grooten et al., 2023)

{{<citation>}}

Bram Grooten, Tristan Tomilin, Gautham Vasan, Matthew E. Taylor, A. Rupam Mahmood, Meng Fang, Mykola Pechenizkiy, Decebal Constantin Mocanu. (2023)  
**MaDi: Learning to Mask Distractions for Generalization in Visual Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15339v1)  

---


**ABSTRACT**  
The visual world provides an abundance of information, but many input pixels received by agents often contain distracting stimuli. Autonomous agents need the ability to distinguish useful information from task-irrelevant perceptions, enabling them to generalize to unseen environments with new distractions. Existing works approach this problem using data augmentation or large auxiliary networks with additional loss functions. We introduce MaDi, a novel algorithm that learns to mask distractions by the reward signal only. In MaDi, the conventional actor-critic structure of deep reinforcement learning agents is complemented by a small third sibling, the Masker. This lightweight neural network generates a mask to determine what the actor and critic will receive, such that they can focus on learning the task. The masks are created dynamically, depending on the current input. We run experiments on the DeepMind Control Generalization Benchmark, the Distracting Control Suite, and a real UR5 Robotic Arm. Our algorithm improves the agent's focus with useful masks, while its efficient Masker network only adds 0.2% more parameters to the original structure, in contrast to previous work. MaDi consistently achieves generalization results better than or competitive to state-of-the-art methods.

{{</citation>}}


### (14/57) Hardware-Aware DNN Compression via Diverse Pruning and Mixed-Precision Quantization (Konstantinos Balaskas et al., 2023)

{{<citation>}}

Konstantinos Balaskas, Andreas Karatzas, Christos Sad, Kostas Siozios, Iraklis Anagnostopoulos, Georgios Zervakis, Jörg Henkel. (2023)  
**Hardware-Aware DNN Compression via Diverse Pruning and Mixed-Precision Quantization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet, Pruning, Quantization, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15322v1)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) have shown significant advantages in a wide variety of domains. However, DNNs are becoming computationally intensive and energy hungry at an exponential pace, while at the same time, there is a vast demand for running sophisticated DNN-based services on resource constrained embedded devices. In this paper, we target energy-efficient inference on embedded DNN accelerators. To that end, we propose an automated framework to compress DNNs in a hardware-aware manner by jointly employing pruning and quantization. We explore, for the first time, per-layer fine- and coarse-grained pruning, in the same DNN architecture, in addition to low bit-width mixed-precision quantization for weights and activations. Reinforcement Learning (RL) is used to explore the associated design space and identify the pruning-quantization configuration so that the energy consumption is minimized whilst the prediction accuracy loss is retained at acceptable levels. Using our novel composite RL agent we are able to extract energy-efficient solutions without requiring retraining and/or fine tuning. Our extensive experimental evaluation over widely used DNNs and the CIFAR-10/100 and ImageNet datasets demonstrates that our framework achieves $39\%$ average energy reduction for $1.7\%$ average accuracy loss and outperforms significantly the state-of-the-art approaches.

{{</citation>}}


### (15/57) Towards Fine-Grained Explainability for Heterogeneous Graph Neural Network (Tong Li et al., 2023)

{{<citation>}}

Tong Li, Jiale Deng, Yanyan Shen, Luyu Qiu, Yongxiang Huang, Caleb Chen Cao. (2023)  
**Towards Fine-Grained Explainability for Heterogeneous Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.15237v1)  

---


**ABSTRACT**  
Heterogeneous graph neural networks (HGNs) are prominent approaches to node classification tasks on heterogeneous graphs. Despite the superior performance, insights about the predictions made from HGNs are obscure to humans. Existing explainability techniques are mainly proposed for GNNs on homogeneous graphs. They focus on highlighting salient graph objects to the predictions whereas the problem of how these objects affect the predictions remains unsolved. Given heterogeneous graphs with complex structures and rich semantics, it is imperative that salient objects can be accompanied with their influence paths to the predictions, unveiling the reasoning process of HGNs. In this paper, we develop xPath, a new framework that provides fine-grained explanations for black-box HGNs specifying a cause node with its influence path to the target node. In xPath, we differentiate the influence of a node on the prediction w.r.t. every individual influence path, and measure the influence by perturbing graph structure via a novel graph rewiring algorithm. Furthermore, we introduce a greedy search algorithm to find the most influential fine-grained explanations efficiently. Empirical results on various HGNs and heterogeneous graphs show that xPath yields faithful explanations efficiently, outperforming the adaptations of advanced GNN explanation approaches.

{{</citation>}}


### (16/57) Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems (Xupeng Miao et al., 2023)

{{<citation>}}

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Hongyi Jin, Tianqi Chen, Zhihao Jia. (2023)  
**Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs-PF, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15234v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of artificial intelligence (AI), generative large language models (LLMs) stand at the forefront, revolutionizing how we interact with our data. However, the computational intensity and memory consumption of deploying these models present substantial challenges in terms of serving efficiency, particularly in scenarios demanding low latency and high throughput. This survey addresses the imperative need for efficient LLM serving methodologies from a machine learning system (MLSys) research perspective, standing at the crux of advanced AI innovations and practical system optimizations. We provide in-depth analysis, covering a spectrum of solutions, ranging from cutting-edge algorithmic modifications to groundbreaking changes in system designs. The survey aims to provide a comprehensive understanding of the current state and future directions in efficient LLM serving, offering valuable insights for researchers and practitioners in overcoming the barriers of effective LLM deployment, thereby reshaping the future of AI.

{{</citation>}}


### (17/57) PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs (Max Zimmer et al., 2023)

{{<citation>}}

Max Zimmer, Megi Andoni, Christoph Spiegel, Sebastian Pokutta. (2023)  
**PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GPT, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2312.15230v1)  

---


**ABSTRACT**  
Neural Networks can be efficiently compressed through pruning, significantly reducing storage and computational demands while maintaining predictive performance. Simple yet effective methods like Iterative Magnitude Pruning (IMP, Han et al., 2015) remove less important parameters and require a costly retraining procedure to recover performance after pruning. However, with the rise of Large Language Models (LLMs), full retraining has become infeasible due to memory and compute constraints. In this study, we challenge the practice of retraining all parameters by demonstrating that updating only a small subset of highly expressive parameters is often sufficient to recover or even improve performance compared to full retraining. Surprisingly, retraining as little as 0.27%-0.35% of the parameters of GPT-architectures (OPT-2.7B/6.7B/13B/30B) achieves comparable performance to One Shot IMP across various sparsity levels. Our method, Parameter-Efficient Retraining after Pruning (PERP), drastically reduces compute and memory demands, enabling pruning and retraining of up to 30 billion parameter models on a single NVIDIA A100 GPU within minutes. Despite magnitude pruning being considered as unsuited for pruning LLMs, our findings show that PERP positions it as a strong contender against state-of-the-art retraining-free approaches such as Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023), opening up a promising alternative to avoiding retraining.

{{</citation>}}


### (18/57) Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It (Federico Siciliano et al., 2023)

{{<citation>}}

Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccin, Irene Amerini, Fabrizio Silvestri. (2023)  
**Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2312.15228v1)  

---


**ABSTRACT**  
Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.

{{</citation>}}


### (19/57) ZO-AdaMU Optimizer: Adapting Perturbation by the Momentum and Uncertainty in Zeroth-order Optimization (Shuoran Jiang et al., 2023)

{{<citation>}}

Shuoran Jiang, Qingcai Chen, Youchen Pan, Yang Xiang, Yukang Lin, Xiangping Wu, Chuanyi Liu, Xiaobao Song. (2023)  
**ZO-AdaMU Optimizer: Adapting Perturbation by the Momentum and Uncertainty in Zeroth-order Optimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.15184v1)  

---


**ABSTRACT**  
Lowering the memory requirement in full-parameter training on large models has become a hot research area. MeZO fine-tunes the large language models (LLMs) by just forward passes in a zeroth-order SGD optimizer (ZO-SGD), demonstrating excellent performance with the same GPU memory usage as inference. However, the simulated perturbation stochastic approximation for gradient estimate in MeZO leads to severe oscillations and incurs a substantial time overhead. Moreover, without momentum regularization, MeZO shows severe over-fitting problems. Lastly, the perturbation-irrelevant momentum on ZO-SGD does not improve the convergence rate. This study proposes ZO-AdaMU to resolve the above problems by adapting the simulated perturbation with momentum in its stochastic approximation. Unlike existing adaptive momentum methods, we relocate momentum on simulated perturbation in stochastic gradient approximation. Our convergence analysis and experiments prove this is a better way to improve convergence stability and rate in ZO-SGD. Extensive experiments demonstrate that ZO-AdaMU yields better generalization for LLMs fine-tuning across various NLP tasks than MeZO and its momentum variants.

{{</citation>}}


### (20/57) Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference (Hongzheng Chen et al., 2023)

{{<citation>}}

Hongzheng Chen, Jiahao Zhang, Yixiao Du, Shaojie Xiang, Zichao Yue, Niansong Zhang, Yaohui Cai, Zhiru Zhang. (2023)  
**Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-AR, cs-CL, cs-LG, cs.LG  
Keywords: BERT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15159v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) boasting billions of parameters have generated a significant demand for efficient deployment in inference workloads. The majority of existing approaches rely on temporal architectures that reuse hardware units for different network layers and operators. However, these methods often encounter challenges in achieving low latency due to considerable memory access overhead. This paper investigates the feasibility and potential of model-specific spatial acceleration for LLM inference on FPGAs. Our approach involves the specialization of distinct hardware units for specific operators or layers, facilitating direct communication between them through a dataflow architecture while minimizing off-chip memory accesses. We introduce a comprehensive analytical model for estimating the performance of a spatial LLM accelerator, taking into account the on-chip compute and memory resources available on an FPGA. Through our analysis, we can determine the scenarios in which FPGA-based spatial acceleration can outperform its GPU-based counterpart. To enable more productive implementations of an LLM model on FPGAs, we further provide a library of high-level synthesis (HLS) kernels that are composable and reusable. This library will be made available as open-source. To validate the effectiveness of both our analytical model and HLS library, we have implemented BERT and GPT2 on an AMD Alveo U280 FPGA device. Experimental results demonstrate our approach can achieve up to 16.1x speedup when compared to previous FPGA-based accelerators for the BERT model. For GPT generative inference, we attain a 2.2x speedup compared to DFX, an FPGA overlay, in the prefill stage, while achieving a 1.9x speedup and a 5.7x improvement in energy efficiency compared to the NVIDIA A100 GPU in the decode stage.

{{</citation>}}


### (21/57) Data Classification With Multiprocessing (Anuja Dixit et al., 2023)

{{<citation>}}

Anuja Dixit, Shreya Byreddy, Guanqun Song, Ting Zhu. (2023)  
**Data Classification With Multiprocessing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15152v1)  

---


**ABSTRACT**  
Classification is one of the most important tasks in Machine Learning (ML) and with recent advancements in artificial intelligence (AI) it is important to find efficient ways to implement it. Generally, the choice of classification algorithm depends on the data it is dealing with, and accuracy of the algorithm depends on the hyperparameters it is tuned with. One way is to check the accuracy of the algorithms by executing it with different hyperparameters serially and then selecting the parameters that give the highest accuracy to predict the final output. This paper proposes another way where the algorithm is parallelly trained with different hyperparameters to reduce the execution time. In the end, results from all the trained variations of the algorithms are ensembled to exploit the parallelism and improve the accuracy of prediction. Python multiprocessing is used to test this hypothesis with different classification algorithms such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), random forest and decision tree and reviews factors affecting parallelism. Ensembled output considers the predictions from all processes and final class is the one predicted by maximum number of processes. Doing this increases the reliability of predictions. We conclude that ensembling improves accuracy and multiprocessing reduces execution time for selected algorithms.

{{</citation>}}


### (22/57) Personalized Federated Learning with Attention-based Client Selection (Zihan Chen et al., 2023)

{{<citation>}}

Zihan Chen, Jundong Li, Cong Shen. (2023)  
**Personalized Federated Learning with Attention-based Client Selection**  

---
Primary Category: cs.LG  
Categories: cs-IT, cs-LG, cs.LG, math-IT  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.15148v1)  

---


**ABSTRACT**  
Personalized Federated Learning (PFL) relies on collective data knowledge to build customized models. However, non-IID data between clients poses significant challenges, as collaborating with clients who have diverse data distributions can harm local model performance, especially with limited training data. To address this issue, we propose FedACS, a new PFL algorithm with an Attention-based Client Selection mechanism. FedACS integrates an attention mechanism to enhance collaboration among clients with similar data distributions and mitigate the data scarcity issue. It prioritizes and allocates resources based on data similarity. We further establish the theoretical convergence behavior of FedACS. Experiments on CIFAR10 and FMNIST validate FedACS's superiority, showcasing its potential to advance personalized federated learning. By tackling non-IID data challenges and data scarcity, FedACS offers promising advances in the field of personalized federated learning.

{{</citation>}}


### (23/57) An FPGA-Based Accelerator for Graph Embedding using Sequential Training Algorithm (Kazuki Sunaga et al., 2023)

{{<citation>}}

Kazuki Sunaga, Keisuke Sugiura, Hiroki Matsutani. (2023)  
**An FPGA-Based Accelerator for Graph Embedding using Sequential Training Algorithm**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.15138v1)  

---


**ABSTRACT**  
A graph embedding is an emerging approach that can represent a graph structure with a fixed-length low-dimensional vector. node2vec is a well-known algorithm to obtain such a graph embedding by sampling neighboring nodes on a given graph with a random walk technique. However, the original node2vec algorithm typically relies on a batch training of graph structures; thus, it is not suited for applications in which the graph structure changes after the deployment. In this paper, we focus on node2vec applications for IoT (Internet of Things) environments. To handle the changes of graph structures after the IoT devices have been deployed in edge environments, in this paper we propose to combine an online sequential training algorithm with node2vec. The proposed sequentially-trainable model is implemented on a resource-limited FPGA (Field-Programmable Gate Array) device to demonstrate the benefits of our approach. The proposed FPGA implementation achieves up to 205.25 times speedup compared to the original model on CPU. Evaluation results using dynamic graphs show that although the original model decreases the accuracy, the proposed sequential model can obtain better graph embedding that can increase the accuracy even when the graph structure is changed.

{{</citation>}}


### (24/57) Gradient Shaping for Multi-Constraint Safe Reinforcement Learning (Yihang Yao et al., 2023)

{{<citation>}}

Yihang Yao, Zuxin Liu, Zhepeng Cen, Peide Huang, Tingnan Zhang, Wenhao Yu, Ding Zhao. (2023)  
**Gradient Shaping for Multi-Constraint Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15127v1)  

---


**ABSTRACT**  
Online safe reinforcement learning (RL) involves training a policy that maximizes task efficiency while satisfying constraints via interacting with the environments. In this paper, our focus lies in addressing the complex challenges associated with solving multi-constraint (MC) safe RL problems. We approach the safe RL problem from the perspective of Multi-Objective Optimization (MOO) and propose a unified framework designed for MC safe RL algorithms. This framework highlights the manipulation of gradients derived from constraints. Leveraging insights from this framework and recognizing the significance of \textit{redundant} and \textit{conflicting} constraint conditions, we introduce the Gradient Shaping (GradS) method for general Lagrangian-based safe RL algorithms to improve the training efficiency in terms of both reward and constraint satisfaction. Our extensive experimentation demonstrates the effectiveness of our proposed method in encouraging exploration and learning a policy that improves both safety and reward performance across various challenging MC safe RL tasks as well as good scalability to the number of constraints.

{{</citation>}}


### (25/57) Scaling Is All You Need: Training Strong Policies for Autonomous Driving with JAX-Accelerated Reinforcement Learning (Moritz Harmel et al., 2023)

{{<citation>}}

Moritz Harmel, Anubhav Paras, Andreas Pasternak, Gary Linscott. (2023)  
**Scaling Is All You Need: Training Strong Policies for Autonomous Driving with JAX-Accelerated Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15122v1)  

---


**ABSTRACT**  
Reinforcement learning has been used to train policies that outperform even the best human players in various games. However, a large amount of data is needed to achieve good performance, which in turn requires building large-scale frameworks and simulators. In this paper, we study how large-scale reinforcement learning can be applied to autonomous driving, analyze how the resulting policies perform as the experiment size is scaled, and what the most important factors contributing to policy performance are. To do this, we first introduce a hardware-accelerated autonomous driving simulator, which allows us to efficiently collect experience from billions of agent steps. This simulator is paired with a large-scale, multi-GPU reinforcement learning framework. We demonstrate that simultaneous scaling of dataset size, model size, and agent steps trained provides increasingly strong driving policies in regard to collision, traffic rule violations, and progress. In particular, our best policy reduces the failure rate by 57% while improving progress by 23% compared to the current state-of-the-art machine learning policies for autonomous driving.

{{</citation>}}


## q-bio.QM (1)



### (26/57) Multimodal Machine Learning Combining Facial Images and Clinical Texts Improves Diagnosis of Rare Genetic Diseases (Da Wu et al., 2023)

{{<citation>}}

Da Wu, Jingye Yang, Steven Klein, Cong Liu, Tzung-Chien Hsieh, Peter Krawitz, Chunhua Weng, Gholson J. Lyon, Jennifer M. Kalish, Kai Wang. (2023)  
**Multimodal Machine Learning Combining Facial Images and Clinical Texts Improves Diagnosis of Rare Genetic Diseases**  

---
Primary Category: q-bio.QM  
Categories: cs-CV, cs-LG, cs-MM, q-bio-GN, q-bio-QM, q-bio.QM  
Keywords: Clinical, Falcon, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15320v1)  

---


**ABSTRACT**  
Individuals with suspected rare genetic disorders often undergo multiple clinical evaluations, imaging studies, laboratory tests and genetic tests, to find a possible answer over a prolonged period of multiple years. Addressing this diagnostic odyssey thus have substantial clinical, psychosocial, and economic benefits. Many rare genetic diseases have distinctive facial features, which can be used by artificial intelligence algorithms to facilitate clinical diagnosis, in prioritizing candidate diseases to be further examined by lab tests or genetic assays, or in helping the phenotype-driven reinterpretation of genome/exome sequencing data. However, existing methods using frontal facial photo were built on conventional Convolutional Neural Networks (CNNs), rely exclusively on facial images, and cannot capture non-facial phenotypic traits and demographic information essential for guiding accurate diagnoses. Here we introduce GestaltMML, a multimodal machine learning (MML) approach solely based on the Transformer architecture. It integrates the facial images, demographic information (age, sex, ethnicity), and clinical notes of patients to improve prediction accuracy. Furthermore, we also introduce GestaltGPT, a GPT-based methodology with few-short learning capacities that exclusively harnesses textual inputs using a range of large language models (LLMs) including Llama 2, GPT-J and Falcon. We evaluated these methods on a diverse range of datasets, including 449 diseases from the GestaltMatcher Database, several in-house datasets on Beckwith-Wiedemann syndrome, Sotos syndrome, NAA10-related syndrome (neurodevelopmental syndrome) and others. Our results suggest that GestaltMML/GestaltGPT effectively incorporate multiple modalities of data, greatly narrow down candidate genetic diagnosis of rare diseases, and may facilitate the reinterpretation of genome/exome sequencing data.

{{</citation>}}


## cs.CL (8)



### (27/57) Paralinguistics-Enhanced Large Language Modeling of Spoken Dialogue (Guan-Ting Lin et al., 2023)

{{<citation>}}

Guan-Ting Lin, Prashanth Gurunath Shivakumar, Ankur Gandhe, Chao-Han Huck Yang, Yile Gu, Shalini Ghosh, Andreas Stolcke, Hung-yi Lee, Ivan Bulyko. (2023)  
**Paralinguistics-Enhanced Large Language Modeling of Spoken Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: BLEU, Dialog, Dialogue, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15316v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated superior abilities in tasks such as chatting, reasoning, and question-answering. However, standard LLMs may ignore crucial paralinguistic information, such as sentiment, emotion, and speaking style, which are essential for achieving natural, human-like spoken conversation, especially when such information is conveyed by acoustic cues. We therefore propose Paralinguistics-enhanced Generative Pretrained Transformer (ParalinGPT), an LLM utilizes text and speech modality to better model the linguistic content and paralinguistic attribute of spoken response. The model takes the conversational context of text, speech embeddings, and paralinguistic attributes as input prompts within a serialized multitasking multi-modal framework. Specifically, our framework serializes tasks in the order of current paralinguistic attribute prediction, response paralinguistic attribute prediction, and response text generation with autoregressive conditioning. We utilize the Switchboard-1 corpus, including its sentiment labels to be the paralinguistic attribute, as our spoken dialogue dataset. Experimental results indicate the proposed serialized multitasking method outperforms typical sequence classification techniques on current and response sentiment classification. Furthermore, leveraging conversational context and speech embeddings significantly improves both response text generation and sentiment prediction. Our proposed framework achieves relative improvements of 6.7%, 12.0%, and 3.5% in current sentiment accuracy, response sentiment accuracy, and response text BLEU score, respectively.

{{</citation>}}


### (28/57) Evaluating the Capability of ChatGPT on Ancient Chinese (Siqing Zhou et al., 2023)

{{<citation>}}

Siqing Zhou, Shijing Si. (2023)  
**Evaluating the Capability of ChatGPT on Ancient Chinese**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.15304v1)  

---


**ABSTRACT**  
ChatGPT's proficiency in handling modern standard languages suggests potential for its use in understanding ancient Chinese.   This project explores ChatGPT's capabilities on ancient Chinese via two tasks: translating ancient Chinese to modern Chinese and recognizing ancient Chinese names. A comparison of ChatGPT's output with human translations serves to evaluate its comprehension of ancient Chinese. The findings indicate that: (1.)the proficiency of ancient Chinese by ChatGPT is yet to reach a satisfactory level; (2.) ChatGPT performs the best on ancient-to-modern translation when feeding with three context sentences. To help reproduce our work, we display the python code snippets used in this study.

{{</citation>}}


### (29/57) Reverse Multi-Choice Dialogue Commonsense Inference with Graph-of-Thought (Li Zheng et al., 2023)

{{<citation>}}

Li Zheng, Hao Fei, Fei Li, Bobo Li, Lizi Liao, Donghong Ji, Chong Teng. (2023)  
**Reverse Multi-Choice Dialogue Commonsense Inference with Graph-of-Thought**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.15291v2)  

---


**ABSTRACT**  
With the proliferation of dialogic data across the Internet, the Dialogue Commonsense Multi-choice Question Answering (DC-MCQ) task has emerged as a response to the challenge of comprehending user queries and intentions. Although prevailing methodologies exhibit effectiveness in addressing single-choice questions, they encounter difficulties in handling multi-choice queries due to the heightened intricacy and informational density. In this paper, inspired by the human cognitive process of progressively excluding options, we propose a three-step Reverse Exclusion Graph-of-Thought (ReX-GoT) framework, including Option Exclusion, Error Analysis, and Combine Information. Specifically, our ReX-GoT mimics human reasoning by gradually excluding irrelevant options and learning the reasons for option errors to choose the optimal path of the GoT and ultimately infer the correct answer. By progressively integrating intricate clues, our method effectively reduces the difficulty of multi-choice reasoning and provides a novel solution for DC-MCQ. Extensive experiments on the CICERO and CICERO$_{v2}$ datasets validate the significant improvement of our approach on DC-MCQ task. On zero-shot setting, our model outperform the best baseline by 17.67% in terms of F1 score for the multi-choice task. Most strikingly, our GPT3.5-based ReX-GoT framework achieves a remarkable 39.44% increase in F1 score.

{{</citation>}}


### (30/57) PokeMQA: Programmable knowledge editing for Multi-hop Question Answering (Hengrui Gu et al., 2023)

{{<citation>}}

Hengrui Gu, Kaixiong Zhou, Xiaotian Han, Ninghao Liu, Ruobing Wang, Xin Wang. (2023)  
**PokeMQA: Programmable knowledge editing for Multi-hop Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.15194v1)  

---


**ABSTRACT**  
Multi-hop question answering (MQA) is one of the challenging tasks to evaluate machine's comprehension and reasoning abilities, where large language models (LLMs) have widely achieved the human-comparable performance. Due to the dynamics of knowledge facts in real world, knowledge editing has been explored to update model with the up-to-date facts while avoiding expensive re-training or fine-tuning. Starting from the edited fact, the updated model needs to provide cascading changes in the chain of MQA. The previous art simply adopts a mix-up prompt to instruct LLMs conducting multiple reasoning tasks sequentially, including question decomposition, answer generation, and conflict checking via comparing with edited facts. However, the coupling of these functionally-diverse reasoning tasks inhibits LLMs' advantages in comprehending and answering questions while disturbing them with the unskilled task of conflict checking. We thus propose a framework, Programmable knowledge editing for Multi-hop Question Answering (PokeMQA), to decouple the jobs. Specifically, we prompt LLMs to decompose knowledge-augmented multi-hop question, while interacting with a detached trainable scope detector to modulate LLMs behavior depending on external conflict signal. The experiments on three LLM backbones and two benchmark datasets validate our superiority in knowledge editing of MQA, outperforming all competitors by a large margin in almost all settings and consistently producing reliable reasoning process.

{{</citation>}}


### (31/57) emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation (Ziyang Ma et al., 2023)

{{<citation>}}

Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, Shiliang Zhang, Xie Chen. (2023)  
**emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs-MM, cs-SD, cs.CL, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.15185v1)  

---


**ABSTRACT**  
We propose emotion2vec, a universal speech emotion representation model. emotion2vec is pre-trained on open-source unlabeled emotion data through self-supervised online distillation, combining utterance-level loss and frame-level loss during pre-training. emotion2vec outperforms state-of-the-art pre-trained universal models and emotion specialist models by only training linear layers for the speech emotion recognition task on the mainstream IEMOCAP dataset. In addition, emotion2vec shows consistent improvements among 10 different languages of speech emotion recognition datasets. emotion2vec also shows excellent results on other emotion tasks, such as song emotion recognition, emotion prediction in conversation, and sentiment analysis. Comparison experiments, ablation experiments, and visualization comprehensively demonstrate the universal capability of the proposed emotion2vec. To the best of our knowledge, emotion2vec is the first universal representation model in various emotion-related tasks, filling a gap in the field.

{{</citation>}}


### (32/57) Multilingual Bias Detection and Mitigation for Indian Languages (Ankita Maity et al., 2023)

{{<citation>}}

Ankita Maity, Anubhav Sharma, Rudra Dhar, Tushar Abhishek, Manish Gupta, Vasudeva Varma. (2023)  
**Multilingual Bias Detection and Mitigation for Indian Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Multilingual, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15181v1)  

---


**ABSTRACT**  
Lack of diverse perspectives causes neutrality bias in Wikipedia content leading to millions of worldwide readers getting exposed by potentially inaccurate information. Hence, neutrality bias detection and mitigation is a critical problem. Although previous studies have proposed effective solutions for English, no work exists for Indian languages. First, we contribute two large datasets, mWikiBias and mWNC, covering 8 languages, for the bias detection and mitigation tasks respectively. Next, we investigate the effectiveness of popular multilingual Transformer-based models for the two tasks by modeling detection as a binary classification problem and mitigation as a style transfer problem. We make the code and data publicly available.

{{</citation>}}


### (33/57) SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling (Dahyun Kim et al., 2023)

{{<citation>}}

Dahyun Kim, Chanjun Park, Sanghoon Kim, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, Sunghun Kim. (2023)  
**SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.15166v1)  

---


**ABSTRACT**  
We introduce depth up-scaling (DUS), a novel technique to up-scale base LLMs efficiently and effectively in a simple manner. In contrast to mixture-of-experts (MoE), DUS does not require complex changes to train and inference. Using DUS, we build SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. Comparative evaluations show that SOLAR 10.7B outperforms existing open-source pretrained LLMs, such as Llama 2 and Mistral 7B. We additionally present SOLAR 10.7B-Instruct, a variant fine-tuned for instruction-following capabilities, surpassing Mixtral-8x7B. SOLAR 10.7B is publicly available under the Apache 2.0 license, promoting broad access and application in the LLM field.

{{</citation>}}


### (34/57) Large Language Models as Zero-Shot Keyphrase Extractor: A Preliminary Empirical Study (Mingyang Song et al., 2023)

{{<citation>}}

Mingyang Song, Xuelian Geng, Songfang Yao, Shilong Lu, Yi Feng, Liping Jing. (2023)  
**Large Language Models as Zero-Shot Keyphrase Extractor: A Preliminary Empirical Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GLM, GPT, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.15156v1)  

---


**ABSTRACT**  
Zero-shot keyphrase extraction aims to build a keyphrase extractor without training by human-annotated data, which is challenging due to the limited human intervention involved. Challenging but worthwhile, zero-shot setting efficiently reduces the time and effort that data labeling takes. Recent efforts on pre-trained large language models (e.g., ChatGPT and ChatGLM) show promising performance on zero-shot settings, thus inspiring us to explore prompt-based methods. In this paper, we ask whether strong keyphrase extraction models can be constructed by directly prompting the large language model ChatGPT. Through experimental results, it is found that ChatGPT still has a lot of room for improvement in the keyphrase extraction task compared to existing state-of-the-art unsupervised and supervised models.

{{</citation>}}


## cs.AI (8)



### (35/57) An Explainable AI Approach to Large Language Model Assisted Causal Model Auditing and Development (Yanming Zhang et al., 2023)

{{<citation>}}

Yanming Zhang, Brette Fitzgibbon, Dino Garofolo, Akshith Kota, Eric Papenhausen, Klaus Mueller. (2023)  
**An Explainable AI Approach to Large Language Model Assisted Causal Model Auditing and Development**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16211v1)  

---


**ABSTRACT**  
Causal networks are widely used in many fields, including epidemiology, social science, medicine, and engineering, to model the complex relationships between variables. While it can be convenient to algorithmically infer these models directly from observational data, the resulting networks are often plagued with erroneous edges. Auditing and correcting these networks may require domain expertise frequently unavailable to the analyst. We propose the use of large language models such as ChatGPT as an auditor for causal networks. Our method presents ChatGPT with a causal network, one edge at a time, to produce insights about edge directionality, possible confounders, and mediating variables. We ask ChatGPT to reflect on various aspects of each causal link and we then produce visualizations that summarize these viewpoints for the human analyst to direct the edge, gather more data, or test further hypotheses. We envision a system where large language models, automated causal inference, and the human analyst and domain expert work hand in hand as a team to derive holistic and comprehensive causal models for any given case scenario. This paper presents first results obtained with an emerging prototype.

{{</citation>}}


### (36/57) Measuring Value Alignment (Fazl Barez et al., 2023)

{{<citation>}}

Fazl Barez, Philip Torr. (2023)  
**Measuring Value Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15241v1)  

---


**ABSTRACT**  
As artificial intelligence (AI) systems become increasingly integrated into various domains, ensuring that they align with human values becomes critical. This paper introduces a novel formalism to quantify the alignment between AI systems and human values, using Markov Decision Processes (MDPs) as the foundational model. We delve into the concept of values as desirable goals tied to actions and norms as behavioral guidelines, aiming to shed light on how they can be used to guide AI decisions. This framework offers a mechanism to evaluate the degree of alignment between norms and values by assessing preference changes across state transitions in a normative world. By utilizing this formalism, AI developers and ethicists can better design and evaluate AI systems to ensure they operate in harmony with human values. The proposed methodology holds potential for a wide range of applications, from recommendation systems emphasizing well-being to autonomous vehicles prioritizing safety.

{{</citation>}}


### (37/57) LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination (Jijia Liu et al., 2023)

{{<citation>}}

Jijia Liu, Chao Yu, Jiaxuan Gao, Yuqing Xie, Qingmin Liao, Yi Wu, Yu Wang. (2023)  
**LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15224v1)  

---


**ABSTRACT**  
AI agents powered by Large Language Models (LLMs) have made significant advances, enabling them to assist humans in diverse complex tasks and leading to a revolution in human-AI coordination. LLM-powered agents typically require invoking LLM APIs and employing artificially designed complex prompts, which results in high inference latency. While this paradigm works well in scenarios with minimal interactive demands, such as code generation, it is unsuitable for highly interactive and real-time applications, such as gaming. Traditional gaming AI often employs small models or reactive policies, enabling fast inference but offering limited task completion and interaction abilities. In this work, we consider Overcooked as our testbed where players could communicate with natural language and cooperate to serve orders. We propose a Hierarchical Language Agent (HLA) for human-AI coordination that provides both strong reasoning abilities while keeping real-time execution. In particular, HLA adopts a hierarchical framework and comprises three modules: a proficient LLM, referred to as Slow Mind, for intention reasoning and language interaction, a lightweight LLM, referred to as Fast Mind, for generating macro actions, and a reactive policy, referred to as Executor, for transforming macro actions into atomic actions. Human studies show that HLA outperforms other baseline agents, including slow-mind-only agents and fast-mind-only agents, with stronger cooperation abilities, faster responses, and more consistent language communications.

{{</citation>}}


### (38/57) Do LLM Agents Exhibit Social Behavior? (Yan Leng et al., 2023)

{{<citation>}}

Yan Leng, Yuan Yuan. (2023)  
**Do LLM Agents Exhibit Social Behavior?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SI, cs.AI, econ-GN, q-fin-EC  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15198v1)  

---


**ABSTRACT**  
The advances of Large Language Models (LLMs) are expanding their utility in both academic research and practical applications. Recent social science research has explored the use of these "black-box" LLM agents for simulating complex social systems and potentially substituting human subjects in experiments. Our study delves into this emerging domain, investigating the extent to which LLMs exhibit key social interaction principles, such as social learning, social preference, and cooperative behavior, in their interactions with humans and other agents. We develop a novel framework for our study, wherein classical laboratory experiments involving human subjects are adapted to use LLM agents. This approach involves step-by-step reasoning that mirrors human cognitive processes and zero-shot learning to assess the innate preferences of LLMs. Our analysis of LLM agents' behavior includes both the primary effects and an in-depth examination of the underlying mechanisms. Focusing on GPT-4, the state-of-the-art LLM, our analyses suggest that LLM agents appear to exhibit a range of human-like social behaviors such as distributional and reciprocity preferences, responsiveness to group identity cues, engagement in indirect reciprocity, and social learning capabilities. However, our analysis also reveals notable differences: LLMs demonstrate a pronounced fairness preference, weaker positive reciprocity, and a more calculating approach in social learning compared to humans. These insights indicate that while LLMs hold great promise for applications in social science research, such as in laboratory experiments and agent-based modeling, the subtle behavioral differences between LLM agents and humans warrant further investigation. Careful examination and development of protocols in evaluating the social behaviors of LLMs are necessary before directly applying these models to emulate human behavior.

{{</citation>}}


### (39/57) Mutual Information as Intrinsic Reward of Reinforcement Learning Agents for On-demand Ride Pooling (Xianjie Zhang et al., 2023)

{{<citation>}}

Xianjie Zhang, Jiahao Sun, Chen Gong, Kai Wang, Yifei Cao, Hao Chen, Hao Chen, Yu Liu. (2023)  
**Mutual Information as Intrinsic Reward of Reinforcement Learning Agents for On-demand Ride Pooling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-SY, cs.AI, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15195v1)  

---


**ABSTRACT**  
The emergence of on-demand ride pooling services allows each vehicle to serve multiple passengers at a time, thus increasing drivers' income and enabling passengers to travel at lower prices than taxi/car on-demand services (only one passenger can be assigned to a car at a time like UberX and Lyft). Although on-demand ride pooling services can bring so many benefits, ride pooling services need a well-defined matching strategy to maximize the benefits for all parties (passengers, drivers, aggregation companies and environment), in which the regional dispatching of vehicles has a significant impact on the matching and revenue. Existing algorithms often only consider revenue maximization, which makes it difficult for requests with unusual distribution to get a ride. How to increase revenue while ensuring a reasonable assignment of requests brings a challenge to ride pooling service companies (aggregation companies). In this paper, we propose a framework for vehicle dispatching for ride pooling tasks, which splits the city into discrete dispatching regions and uses the reinforcement learning (RL) algorithm to dispatch vehicles in these regions. We also consider the mutual information (MI) between vehicle and order distribution as the intrinsic reward of the RL algorithm to improve the correlation between their distributions, thus ensuring the possibility of getting a ride for unusually distributed requests. In experimental results on a real-world taxi dataset, we demonstrate that our framework can significantly increase revenue up to an average of 3\% over the existing best on-demand ride pooling method.

{{</citation>}}


### (40/57) Reinforcement Learning for Safe Occupancy Strategies in Educational Spaces during an Epidemic (Elizabeth Akinyi Ondula et al., 2023)

{{<citation>}}

Elizabeth Akinyi Ondula, Bhaskar Krishnamachari. (2023)  
**Reinforcement Learning for Safe Occupancy Strategies in Educational Spaces during an Epidemic**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15163v1)  

---


**ABSTRACT**  
Epidemic modeling, encompassing deterministic and stochastic approaches, is vital for understanding infectious diseases and informing public health strategies. This research adopts a prescriptive approach, focusing on reinforcement learning (RL) to develop strategies that balance minimizing infections with maximizing in-person interactions in educational settings. We introduce SafeCampus , a novel tool that simulates infection spread and facilitates the exploration of various RL algorithms in response to epidemic challenges. SafeCampus incorporates a custom RL environment, informed by stochastic epidemic models, to realistically represent university campus dynamics during epidemics. We evaluate Q-learning for a discretized state space which resulted in a policy matrix that not only guides occupancy decisions under varying epidemic conditions but also illustrates the inherent trade-off in epidemic management. This trade-off is characterized by the dilemma between stricter measures, which may effectively reduce infections but impose less educational benefit (more in-person interactions), and more lenient policies, which could lead to higher infection rates.

{{</citation>}}


### (41/57) Networks of Classical Conditioning Gates and Their Learning (Shun-ichi Azuma et al., 2023)

{{<citation>}}

Shun-ichi Azuma, Dai Takakura, Ryo Ariizumi, Toru Asai. (2023)  
**Networks of Classical Conditioning Gates and Their Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LO, cs-NE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15161v1)  

---


**ABSTRACT**  
Chemical AI is chemically synthesized artificial intelligence that has the ability of learning in addition to information processing. A research project on chemical AI, called the Molecular Cybernetics Project, was launched in Japan in 2021 with the goal of creating a molecular machine that can learn a type of conditioned reflex through the process called classical conditioning. If the project succeeds in developing such a molecular machine, the next step would be to configure a network of such machines to realize more complex functions. With this motivation, this paper develops a method for learning a desired function in the network of nodes each of which can implement classical conditioning. First, we present a model of classical conditioning, which is called here a classical conditioning gate. We then propose a learning algorithm for the network of classical conditioning gates.

{{</citation>}}


### (42/57) Human-AI Collaboration in Real-World Complex Environment with Reinforcement Learning (Md Saiful Islam et al., 2023)

{{<citation>}}

Md Saiful Islam, Srijita Das, Sai Krishna Gottipati, William Duguay, Clodéric Mars, Jalal Arabneydi, Antoine Fagette, Matthew Guzdial, Matthew-E-Taylor. (2023)  
**Human-AI Collaboration in Real-World Complex Environment with Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs-MA, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15160v1)  

---


**ABSTRACT**  
Recent advances in reinforcement learning (RL) and Human-in-the-Loop (HitL) learning have made human-AI collaboration easier for humans to team with AI agents. Leveraging human expertise and experience with AI in intelligent systems can be efficient and beneficial. Still, it is unclear to what extent human-AI collaboration will be successful, and how such teaming performs compared to humans or AI agents only. In this work, we show that learning from humans is effective and that human-AI collaboration outperforms human-controlled and fully autonomous AI agents in a complex simulation environment. In addition, we have developed a new simulator for critical infrastructure protection, focusing on a scenario where AI-powered drones and human teams collaborate to defend an airport against enemy drone attacks. We develop a user interface to allow humans to assist AI agents effectively. We demonstrated that agents learn faster while learning from policy correction compared to learning from humans or agents. Furthermore, human-AI collaboration requires lower mental and temporal demands, reduces human effort, and yields higher performance than if humans directly controlled all agents. In conclusion, we show that humans can provide helpful advice to the RL agents, allowing them to improve learning in a multi-agent setting.

{{</citation>}}


## cs.IR (2)



### (43/57) Monitoring the Evolution of Behavioural Embeddings in Social Media Recommendation (Srijan Saket, 2023)

{{<citation>}}

Srijan Saket. (2023)  
**Monitoring the Evolution of Behavioural Embeddings in Social Media Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Embedding, Social Media  
[Paper Link](http://arxiv.org/abs/2312.15265v1)  

---


**ABSTRACT**  
Short video applications pose unique challenges for recommender systems due to the constant influx of new content and the absence of historical user interactions for quality assessment of uploaded content. This research characterizes the evolution of embeddings in short video recommendation systems, comparing batch and real-time updates to content embeddings. The analysis investigates embedding maturity, the learning peak during view accumulation, popularity bias, l2-norm distribution of learned embeddings, and their impact on user engagement metrics. The study unveils the contrast in the number of interactions needed to achieve mature embeddings in both learning modes, identifies the ideal learning point, and explores the distribution of l2-norm across various update methods. Utilizing a production system deployed on a large-scale short video app with over 180 million users, the findings offer insights into designing effective recommendation systems and enhancing user satisfaction and engagement in short video applications.

{{</citation>}}


### (44/57) Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns (Xin Liu et al., 2023)

{{<citation>}}

Xin Liu, Zheng Li, Yifan Gao, Jingfeng Yang, Tianyu Cao, Zhengyang Wang, Bing Yin, Yangqiu Song. (2023)  
**Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.16199v1)  

---


**ABSTRACT**  
The goal of session-based recommendation in E-commerce is to predict the next item that an anonymous user will purchase based on the browsing and purchase history. However, constructing global or local transition graphs to supplement session data can lead to noisy correlations and user intent vanishing. In this work, we propose the Frequent Attribute Pattern Augmented Transformer (FAPAT) that characterizes user intents by building attribute transition graphs and matching attribute patterns. Specifically, the frequent and compact attribute patterns are served as memory to augment session representations, followed by a gate and a transformer block to fuse the whole session information. Through extensive experiments on two public benchmarks and 100 million industrial data in three domains, we demonstrate that FAPAT consistently outperforms state-of-the-art methods by an average of 4.5% across various evaluation metrics (Hits, NDCG, MRR). Besides evaluating the next-item prediction, we estimate the models' capabilities to capture user intents via predicting items' attributes and period-item recommendations.

{{</citation>}}


## cs.CR (3)



### (45/57) A Security Enhanced Authentication Protocol (Sai Sreekar Vankayalapati et al., 2023)

{{<citation>}}

Sai Sreekar Vankayalapati, Srijanee Mookherji, Vanga Odelu. (2023)  
**A Security Enhanced Authentication Protocol**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.15250v1)  

---


**ABSTRACT**  
Internet of Things (IoT) have gained popularity in recent times. With an increase in the number of IoT devices, security and privacy vulnerabilities are also increasing. For sensitive domains like healthcare and industrial sectors, such vulnerabilities can cause havoc. Thus, authentication is an important aspect for establishing a secure communication between various participants. In this paper, we study the two recent authentication and key exchange protocols. We prove that these protocols are vulnerable to replay attack and modification attack, and also suffer from technical correctness. We then present the possible improvements to overcome the discussed vulnerabilities. The enhancement preserves performance of the original protocols.

{{</citation>}}


### (46/57) Security in 5G Networks -- How 5G networks help Mitigate Location Tracking Vulnerability (Abshir Ali et al., 2023)

{{<citation>}}

Abshir Ali, Guanqun Song, Ting Zhu. (2023)  
**Security in 5G Networks -- How 5G networks help Mitigate Location Tracking Vulnerability**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.16200v1)  

---


**ABSTRACT**  
As 5G networks become more mainstream, privacy has come to the forefront of end users. More scrutiny has been shown to previous generation cellular technologies such as 3G and 4G on how they handle sensitive metadata transmitted from an end user mobile device to base stations during registration with a cellular network. These generation cellular networks do not enforce any encryption on this information transmitted during this process, giving malicious actors an easy way to intercept the information. Such an interception can allow an adversary to locate end users with shocking accuracy. This paper investigates this problem in great detail and discusses how a newly introduced approach in 5G networks is helping combat this problem. The paper discusses the implications of this vulnerability and the technical details of the new approach, including the encryption schemes used to secure this sensitive information. Finally, the paper will discuss any limitations to this new approach.

{{</citation>}}


### (47/57) The Inner Workings of Windows Security (Ashvini A Kulshrestha et al., 2023)

{{<citation>}}

Ashvini A Kulshrestha, Guanqun Song, Ting Zhu. (2023)  
**The Inner Workings of Windows Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Microsoft, Security  
[Paper Link](http://arxiv.org/abs/2312.15150v1)  

---


**ABSTRACT**  
The year 2022 saw a significant increase in Microsoft vulnerabilities, reaching an all-time high in the past decade. With new vulnerabilities constantly emerging, there is an urgent need for proactive approaches to harden systems and protect them from potential cyber threats. This project aims to investigate the vulnerabilities of the Windows Operating System and explore the effectiveness of key security features such as BitLocker, Microsoft Defender, and Windows Firewall in addressing these threats. To achieve this, various security threats are simulated in controlled environments using coded examples, allowing for a thorough evaluation of the security solutions' effectiveness. Based on the results, this study will provide recommendations for mitigation strategies to enhance system security and strengthen the protection provided by Windows security features. By identifying potential weaknesses and areas of improvement in the Windows security infrastructure, this project will contribute to the development of more robust and resilient security solutions that can better safeguard systems against emerging cyber threats.

{{</citation>}}


## cs.MM (1)



### (48/57) QoE modeling for Voice over IP: Simplified E-model Enhancement Utilizing the Subjective MOS Prediction Model (Therdpong Daengsi et al., 2023)

{{<citation>}}

Therdpong Daengsi, Pongpisit Wuttidittachotti. (2023)  
**QoE modeling for Voice over IP: Simplified E-model Enhancement Utilizing the Subjective MOS Prediction Model**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.15239v1)  

---


**ABSTRACT**  
This research proposes an enhanced measurement method for VoIP quality assessment which provides an improvement to accuracy and reliability. To improve the objective measurement tool called the simplified E-model for the selected codec, G.729, it has been enhanced by utilizing a subjective MOS prediction model based on native Thai users, who use the Thai-tonal language. Then, the different results from the simplified E-model and subjective MOS prediction model were used to create the Bias function, before adding to the simplified E-model. Finally, it has been found that the outputs from the enhanced simplified E-model for the G.729 codec shows better accuracy when compared to the original simplified E-model, specially, after the enhanced model has been evaluated with 4 test sets. The major contribution of this enhancement is that errors are reduced by 58.87 % when compared to the generic simplified E-model. That means the enhanced simplified E-model as proposed in this study can provide improvement beyond the original simplified one significantly.

{{</citation>}}


## cs.CE (1)



### (49/57) MASTER: Market-Guided Stock Transformer for Stock Price Forecasting (Tong Li et al., 2023)

{{<citation>}}

Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, Sen Huang. (2023)  
**MASTER: Market-Guided Stock Transformer for Stock Price Forecasting**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.15235v1)  

---


**ABSTRACT**  
Stock price forecasting has remained an extremely challenging problem for many decades due to the high volatility of the stock market. Recent efforts have been devoted to modeling complex stock correlations toward joint stock price forecasting. Existing works share a common neural architecture that learns temporal patterns from individual stock series and then mixes up temporal representations to establish stock correlations. However, they only consider time-aligned stock correlations stemming from all the input stock features, which suffer from two limitations. First, stock correlations often occur momentarily and in a cross-time manner. Second, the feature effectiveness is dynamic with market variation, which affects both the stock sequential patterns and their correlations. To address the limitations, this paper introduces MASTER, a MArkert-Guided Stock TransformER, which models the momentary and cross-time stock correlation and leverages market information for automatic feature selection. MASTER elegantly tackles the complex stock correlation by alternatively engaging in intra-stock and inter-stock information aggregation. Experiments show the superiority of MASTER compared with previous works and visualize the captured realistic stock correlation to provide valuable insights.

{{</citation>}}


## cs.SE (3)



### (50/57) A Survey on Large Language Models for Software Engineering (Quanjun Zhang et al., 2023)

{{<citation>}}

Quanjun Zhang, Chunrong Fang, Yang Xie, Yaxin Zhang, Yun Yang, Weisong Sun, Shengcheng Yu, Zhenyu Chen. (2023)  
**A Survey on Large Language Models for Software Engineering**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15223v1)  

---


**ABSTRACT**  
Software Engineering (SE) is the systematic design, development, and maintenance of software applications, underpinning the digital infrastructure of our modern mainworld. Very recently, the SE community has seen a rapidly increasing number of techniques employing Large Language Models (LLMs) to automate a broad range of SE tasks. Nevertheless, existing information of the applications, effects, and possible limitations of LLMs within SE is still not well-studied.   In this paper, we provide a systematic survey to summarize the current state-of-the-art research in the LLM-based SE community. We summarize 30 representative LLMs of Source Code across three model architectures, 15 pre-training objectives across four categories, and 16 downstream tasks across five categories. We then present a detailed summarization of the recent SE studies for which LLMs are commonly utilized, including 155 studies for 43 specific code-related tasks across four crucial phases within the SE workflow. Besides, we summarize existing attempts to empirically evaluate LLMs in SE, such as benchmarks, empirical studies, and exploration of SE education. We also discuss several critical aspects of optimization and applications of LLMs in SE, such as security attacks, model tuning, and model compression. Finally, we highlight several challenges and potential opportunities on applying LLMs for future SE studies, such as exploring domain LLMs and constructing clean evaluation datasets. Overall, our work can help researchers gain a comprehensive understanding about the achievements of the existing LLM-based SE studies and promote the practical application of these techniques. Our artifacts are publicly available and will continuously updated at the living repository: \url{https://github.com/iSEngLab/AwesomeLLM4SE}.

{{</citation>}}


### (51/57) Enhancing Code Intelligence Tasks with ChatGPT (Kang Yang et al., 2023)

{{<citation>}}

Kang Yang, Xinjun Mao, Shangwen Wang, Tanghaoran Zhang, Bo Lin, Yanlin Wang, Yihao Qin, Zhang Zhang, Xiaoguang Mao. (2023)  
**Enhancing Code Intelligence Tasks with ChatGPT**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, T5  
[Paper Link](http://arxiv.org/abs/2312.15202v1)  

---


**ABSTRACT**  
Pre-trained code models have emerged as crucial tools in various code intelligence tasks. However, their effectiveness depends on the quality of the pre-training dataset, particularly the human reference comments, which serve as a bridge between the programming language and natural language. One significant challenge is that such comments can become inconsistent with the corresponding code as the software evolves. This discrepancy can lead to suboptimal training of the models, decreasing their performances. LLMs have demonstrated superior capabilities in generating high-quality code comments. In light of that, we try to tackle the quality issue of the dataset by harnessing the power of LLMs. Specifically, we raise the question: Can we rebuild the pre-training dataset by substituting the original comments with LLM-generated ones for more effective pre-trained code models? To answer the question, we first conduct a comprehensive evaluation to compare ChatGPT-generated comments with human reference comments. As existing reference-based metrics treat the reference comments as gold standards, we introduce two auxiliary tasks as novel reference-free metrics to assess the quality of comments, i.e., code-comment inconsistency detection and code search. Experimental results show that ChatGPT-generated comments demonstrate superior semantic consistency with the code compared to human references, indicating the potential of utilizing ChatGPT to enhance the quality of the pre-training dataset. We rebuilt the widely used dataset, CodeSearchNet, with ChatGPT-generated comments. Subsequent experiments involve re-pre-training the CodeT5 with our refined dataset.Evaluation results on four generation tasks and one understanding code intelligence tasks show that the model pre-trained by ChatGPT-enhanced data outperforms its counterpart on code summarization, code generation, and code translation tasks.

{{</citation>}}


### (52/57) CodeScholar: Growing Idiomatic Code Examples (Manish Shetty et al., 2023)

{{<citation>}}

Manish Shetty, Koushik Sen, Ion Stoica. (2023)  
**CodeScholar: Growing Idiomatic Code Examples**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-PL, cs-SE, cs.SE  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.15157v1)  

---


**ABSTRACT**  
Programmers often search for usage examples for API methods. A tool that could generate realistic, idiomatic, and contextual usage examples for one or more APIs would be immensely beneficial to developers. Such a tool would relieve the need for a deep understanding of the API landscape, augment existing documentation, and help discover interactions among APIs. We present CodeScholar, a tool that generates idiomatic code examples demonstrating the common usage of API methods. It includes a novel neural-guided search technique over graphs that grows the query APIs into idiomatic code examples. Our user study demonstrates that in 70% of cases, developers prefer CodeScholar generated examples over state-of-the-art large language models (LLM) like GPT3.5. We quantitatively evaluate 60 single and 25 multi-API queries from 6 popular Python libraries and show that across-the-board CodeScholar generates more realistic, diverse, and concise examples. In addition, we show that CodeScholar not only helps developers but also LLM-powered programming assistants generate correct code in a program synthesis setting.

{{</citation>}}


## cs.SD (2)



### (53/57) TransFace: Unit-Based Audio-Visual Speech Synthesizer for Talking Head Translation (Xize Cheng et al., 2023)

{{<citation>}}

Xize Cheng, Rongjie Huang, Linjun Li, Tao Jin, Zehan Wang, Aoxiong Yin, Minglei Li, Xinyu Duan, changpeng yang, Zhou Zhao. (2023)  
**TransFace: Unit-Based Audio-Visual Speech Synthesizer for Talking Head Translation**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-CV, cs-SD, cs.SD, eess-AS  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2312.15197v1)  

---


**ABSTRACT**  
Direct speech-to-speech translation achieves high-quality results through the introduction of discrete units obtained from self-supervised learning. This approach circumvents delays and cascading errors associated with model cascading. However, talking head translation, converting audio-visual speech (i.e., talking head video) from one language into another, still confronts several challenges compared to audio speech: (1) Existing methods invariably rely on cascading, synthesizing via both audio and text, resulting in delays and cascading errors. (2) Talking head translation has a limited set of reference frames. If the generated translation exceeds the length of the original speech, the video sequence needs to be supplemented by repeating frames, leading to jarring video transitions. In this work, we propose a model for talking head translation, \textbf{TransFace}, which can directly translate audio-visual speech into audio-visual speech in other languages. It consists of a speech-to-unit translation model to convert audio speech into discrete units and a unit-based audio-visual speech synthesizer, Unit2Lip, to re-synthesize synchronized audio-visual speech from discrete units in parallel. Furthermore, we introduce a Bounded Duration Predictor, ensuring isometric talking head translation and preventing duplicate reference frames. Experiments demonstrate that our proposed Unit2Lip model significantly improves synchronization (1.601 and 0.982 on LSE-C for the original and generated audio speech, respectively) and boosts inference speed by a factor of 4.35 on LRS2. Additionally, TransFace achieves impressive BLEU scores of 61.93 and 47.55 for Es-En and Fr-En on LRS3-T and 100% isochronous translations.

{{</citation>}}


### (54/57) SAIC: Integration of Speech Anonymization and Identity Classification (Ming Cheng et al., 2023)

{{<citation>}}

Ming Cheng, Xingjian Diao, Shitong Cheng, Wenjun Liu. (2023)  
**SAIC: Integration of Speech Anonymization and Identity Classification**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CR, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15190v1)  

---


**ABSTRACT**  
Speech anonymization and de-identification have garnered significant attention recently, especially in the healthcare area including telehealth consultations, patient voiceprint matching, and patient real-time monitoring. Speaker identity classification tasks, which involve recognizing specific speakers from audio to learn identity features, are crucial for de-identification. Since rare studies have effectively combined speech anonymization with identity classification, we propose SAIC - an innovative pipeline for integrating Speech Anonymization and Identity Classification. SAIC demonstrates remarkable performance and reaches state-of-the-art in the speaker identity classification task on the Voxceleb1 dataset, with a top-1 accuracy of 96.1%. Although SAIC is not trained or evaluated specifically on clinical data, the result strongly proves the model's effectiveness and the possibility to generalize into the healthcare area, providing insightful guidance for future work.

{{</citation>}}


## cs.DC (1)



### (55/57) Efficient Asynchronous Federated Learning with Sparsification and Quantization (Juncheng Jia et al., 2023)

{{<citation>}}

Juncheng Jia, Ji Liu, Chendi Zhou, Hao Tian, Mianxiong Dong, Dejing Dou. (2023)  
**Efficient Asynchronous Federated Learning with Sparsification and Quantization**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs-LG, cs.DC  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.15186v1)  

---


**ABSTRACT**  
While data is distributed in multiple edge devices, Federated Learning (FL) is attracting more and more attention to collaboratively train a machine learning model without transferring raw data. FL generally exploits a parameter server and a large number of edge devices during the whole process of the model training, while several devices are selected in each round. However, straggler devices may slow down the training process or even make the system crash during training. Meanwhile, other idle edge devices remain unused. As the bandwidth between the devices and the server is relatively low, the communication of intermediate data becomes a bottleneck. In this paper, we propose Time-Efficient Asynchronous federated learning with Sparsification and Quantization, i.e., TEASQ-Fed. TEASQ-Fed can fully exploit edge devices to asynchronously participate in the training process by actively applying for tasks. We utilize control parameters to choose an appropriate number of parallel edge devices, which simultaneously execute the training tasks. In addition, we introduce a caching mechanism and weighted averaging with respect to model staleness to further improve the accuracy. Furthermore, we propose a sparsification and quantitation approach to compress the intermediate data to accelerate the training. The experimental results reveal that TEASQ-Fed improves the accuracy (up to 16.67% higher) while accelerating the convergence of model training (up to twice faster).

{{</citation>}}


## eess.IV (1)



### (56/57) Narrowing the semantic gaps in U-Net with learnable skip connections: The case of medical image segmentation (Haonan Wang et al., 2023)

{{<citation>}}

Haonan Wang, Peng Cao, Xiaoli Liu, Jinzhu Yang, Osmar Zaiane. (2023)  
**Narrowing the semantic gaps in U-Net with learnable skip connections: The case of medical image segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15182v1)  

---


**ABSTRACT**  
Most state-of-the-art methods for medical image segmentation adopt the encoder-decoder architecture. However, this U-shaped framework still has limitations in capturing the non-local multi-scale information with a simple skip connection. To solve the problem, we firstly explore the potential weakness of skip connections in U-Net on multiple segmentation tasks, and find that i) not all skip connections are useful, each skip connection has different contribution; ii) the optimal combinations of skip connections are different, relying on the specific datasets. Based on our findings, we propose a new segmentation framework, named UDTransNet, to solve three semantic gaps in U-Net. Specifically, we propose a Dual Attention Transformer (DAT) module for capturing the channel- and spatial-wise relationships to better fuse the encoder features, and a Decoder-guided Recalibration Attention (DRA) module for effectively connecting the DAT tokens and the decoder features to eliminate the inconsistency. Hence, both modules establish a learnable connection to solve the semantic gaps between the encoder and the decoder, which leads to a high-performance segmentation model for medical images. Comprehensive experimental results indicate that our UDTransNet produces higher evaluation scores and finer segmentation results with relatively fewer parameters over the state-of-the-art segmentation methods on different public datasets. Code: https://github.com/McGregorWwww/UDTransNet.

{{</citation>}}


## cs.SI (1)



### (57/57) Majority-based Preference Diffusion on Social Networks (Ahad N. Zehmakan, 2023)

{{<citation>}}

Ahad N. Zehmakan. (2023)  
**Majority-based Preference Diffusion on Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-DS, cs-MA, cs-SI, cs.SI, math-CO, physics-soc-ph  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.15140v1)  

---


**ABSTRACT**  
We study a majority based preference diffusion model in which the members of a social network update their preferences based on those of their connections. Consider an undirected graph where each node has a strict linear order over a set of $\alpha$ alternatives. At each round, a node randomly selects two adjacent alternatives and updates their relative order with the majority view of its neighbors. We bound the convergence time of the process in terms of the number of nodes/edges and $\alpha$. Furthermore, we study the minimum cost to ensure that a desired alternative will ``win'' the process, where occupying each position in a preference order of a node has a cost. We prove tight bounds on the minimum cost for general graphs and graphs with strong expansion properties. Furthermore, we investigate a more light-weight process where each node chooses one of its neighbors uniformly at random and copies its order fully with some fixed probability and remains unchanged otherwise. We characterize the convergence properties of this process, namely convergence time and stable states, using Martingale and reversible Markov chain analysis. Finally, we present the outcomes of our experiments conducted on different synthetic random graph models and graph data from online social platforms. These experiments not only support our theoretical findings, but also shed some light on some other fundamental problems, such as designing powerful countermeasures.

{{</citation>}}
