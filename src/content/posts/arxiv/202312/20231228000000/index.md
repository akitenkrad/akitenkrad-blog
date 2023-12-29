---
draft: false
title: "arXiv @ 2023.12.28"
date: 2023-12-28
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.28"
    identifier: arxiv_20231228
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (19)](#cscv-19)
- [cs.CL (21)](#cscl-21)
- [cs.LG (13)](#cslg-13)
- [cs.IR (6)](#csir-6)
- [cs.NI (1)](#csni-1)
- [cs.AI (3)](#csai-3)
- [cs.RO (2)](#csro-2)
- [cs.CY (1)](#cscy-1)
- [cs.SE (1)](#csse-1)

## cs.CV (19)



### (1/67) Universal Pyramid Adversarial Training for Improved ViT Performance (Ping-yeh Chiang et al., 2023)

{{<citation>}}

Ping-yeh Chiang, Yipin Zhou, Omid Poursaeed, Satya Narayan Shukla, Ashish Shah, Tom Goldstein, Ser-Nam Lim. (2023)  
**Universal Pyramid Adversarial Training for Improved ViT Performance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2312.16339v1)  

---


**ABSTRACT**  
Recently, Pyramid Adversarial training (Herrmann et al., 2022) has been shown to be very effective for improving clean accuracy and distribution-shift robustness of vision transformers. However, due to the iterative nature of adversarial training, the technique is up to 7 times more expensive than standard training. To make the method more efficient, we propose Universal Pyramid Adversarial training, where we learn a single pyramid adversarial pattern shared across the whole dataset instead of the sample-wise patterns. With our proposed technique, we decrease the computational cost of Pyramid Adversarial training by up to 70% while retaining the majority of its benefit on clean performance and distribution-shift robustness. In addition, to the best of our knowledge, we are also the first to find that universal adversarial training can be leveraged to improve clean model performance.

{{</citation>}}


### (2/67) State-of-the-Art in Nudity Classification: A Comparative Analysis (Fatih Cagatay Akyon et al., 2023)

{{<citation>}}

Fatih Cagatay Akyon, Alptekin Temizel. (2023)  
**State-of-the-Art in Nudity Classification: A Comparative Analysis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16338v1)  

---


**ABSTRACT**  
This paper presents a comparative analysis of existing nudity classification techniques for classifying images based on the presence of nudity, with a focus on their application in content moderation. The evaluation focuses on CNN-based models, vision transformer, and popular open-source safety checkers from Stable Diffusion and Large-scale Artificial Intelligence Open Network (LAION). The study identifies the limitations of current evaluation datasets and highlights the need for more diverse and challenging datasets. The paper discusses the potential implications of these findings for developing more accurate and effective image classification systems on online platforms. Overall, the study emphasizes the importance of continually improving image classification models to ensure the safety and well-being of platform users. The project page, including the demonstrations and results is publicly available at https://github.com/fcakyon/content-moderation-deep-learning.

{{</citation>}}


### (3/67) EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI (Tai Wang et al., 2023)

{{<citation>}}

Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, Xihui Liu, Cewu Lu, Dahua Lin, Jiangmiao Pang. (2023)  
**EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16170v1)  

---


**ABSTRACT**  
In the realm of computer vision and robotics, embodied agents are expected to explore their environment and carry out human instructions. This necessitates the ability to fully understand 3D scenes given their first-person observations and contextualize them into language for interaction. However, traditional research focuses more on scene-level input and output setups from a global view. To address the gap, we introduce EmbodiedScan, a multi-modal, ego-centric 3D perception dataset and benchmark for holistic 3D scene understanding. It encompasses over 5k scans encapsulating 1M ego-centric RGB-D views, 1M language prompts, 160k 3D-oriented boxes spanning over 760 categories, some of which partially align with LVIS, and dense semantic occupancy with 80 common categories. Building upon this database, we introduce a baseline framework named Embodied Perceptron. It is capable of processing an arbitrary number of multi-modal inputs and demonstrates remarkable 3D perception capabilities, both within the two series of benchmarks we set up, i.e., fundamental 3D perception tasks and language-grounded tasks, and in the wild. Codes, datasets, and benchmarks will be available at https://github.com/OpenRobotLab/EmbodiedScan.

{{</citation>}}


### (4/67) Social-Transmotion: Promptable Human Trajectory Prediction (Saeed Saadatnejad et al., 2023)

{{<citation>}}

Saeed Saadatnejad, Yang Gao, Kaouther Messaoud, Alexandre Alahi. (2023)  
**Social-Transmotion: Promptable Human Trajectory Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.16168v1)  

---


**ABSTRACT**  
Accurate human trajectory prediction is crucial for applications such as autonomous vehicles, robotics, and surveillance systems. Yet, existing models often fail to fully leverage the non-verbal social cues human subconsciously communicate when navigating the space. To address this, we introduce Social-Transmotion, a generic model that exploits the power of transformers to handle diverse and numerous visual cues, capturing the multi-modal nature of human behavior. We translate the idea of a prompt from Natural Language Processing (NLP) to the task of human trajectory prediction, where a prompt can be a sequence of x-y coordinates on the ground, bounding boxes or body poses. This, in turn, augments trajectory data, leading to enhanced human trajectory prediction. Our model exhibits flexibility and adaptability by capturing spatiotemporal interactions between pedestrians based on the available visual cues, whether they are poses, bounding boxes, or a combination thereof. By the masking technique, we ensure our model's effectiveness even when certain visual cues are unavailable, although performance is further boosted with the presence of comprehensive visual data. We delve into the merits of using 2d versus 3d poses, and a limited set of poses. Additionally, we investigate the spatial and temporal attention map to identify which keypoints and frames of poses are vital for optimizing human trajectory prediction. Our approach is validated on multiple datasets, including JTA, JRDB, Pedestrians and Cyclists in Road Traffic, and ETH-UCY. The code is publicly available: https://github.com/vita-epfl/social-transmotion

{{</citation>}}


### (5/67) Cloud-Device Collaborative Learning for Multimodal Large Language Models (Guanqun Wang et al., 2023)

{{<citation>}}

Guanqun Wang, Jiaming Liu, Chenxuan Li, Junpeng Ma, Yuan Zhang, Xinyu Wei, Kevin Zhang, Maurice Chong, Ray Zhang, Yijiang Liu, Shanghang Zhang. (2023)  
**Cloud-Device Collaborative Learning for Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16279v1)  

---


**ABSTRACT**  
The burgeoning field of Multimodal Large Language Models (MLLMs) has exhibited remarkable performance in diverse tasks such as captioning, commonsense reasoning, and visual scene understanding. However, the deployment of these large-scale MLLMs on client devices is hindered by their extensive model parameters, leading to a notable decline in generalization capabilities when these models are compressed for device deployment. Addressing this challenge, we introduce a Cloud-Device Collaborative Continual Adaptation framework, designed to enhance the performance of compressed, device-deployed MLLMs by leveraging the robust capabilities of cloud-based, larger-scale MLLMs. Our framework is structured into three key components: a device-to-cloud uplink for efficient data transmission, cloud-based knowledge adaptation, and an optimized cloud-to-device downlink for model deployment. In the uplink phase, we employ an Uncertainty-guided Token Sampling (UTS) strategy to effectively filter out-of-distribution tokens, thereby reducing transmission costs and improving training efficiency. On the cloud side, we propose Adapter-based Knowledge Distillation (AKD) method to transfer refined knowledge from large-scale to compressed, pocket-size MLLMs. Furthermore, we propose a Dynamic Weight update Compression (DWC) strategy for the downlink, which adaptively selects and quantizes updated weight parameters, enhancing transmission efficiency and reducing the representational disparity between cloud and device models. Extensive experiments on several multimodal benchmarks demonstrate the superiority of our proposed framework over prior Knowledge Distillation and device-cloud collaboration methods. Notably, we also validate the feasibility of our approach to real-world experiments.

{{</citation>}}


### (6/67) VirtualPainting: Addressing Sparsity with Virtual Points and Distance-Aware Data Augmentation for 3D Object Detection (Sudip Dhakal et al., 2023)

{{<citation>}}

Sudip Dhakal, Dominic Carrillo, Deyuan Qu, Michael Nutt, Qing Yang, Song Fu. (2023)  
**VirtualPainting: Addressing Sparsity with Virtual Points and Distance-Aware Data Augmentation for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.16141v1)  

---


**ABSTRACT**  
In recent times, there has been a notable surge in multimodal approaches that decorates raw LiDAR point clouds with camera-derived features to improve object detection performance. However, we found that these methods still grapple with the inherent sparsity of LiDAR point cloud data, primarily because fewer points are enriched with camera-derived features for sparsely distributed objects. We present an innovative approach that involves the generation of virtual LiDAR points using camera images and enhancing these virtual points with semantic labels obtained from image-based segmentation networks to tackle this issue and facilitate the detection of sparsely distributed objects, particularly those that are occluded or distant. Furthermore, we integrate a distance aware data augmentation (DADA) technique to enhance the models capability to recognize these sparsely distributed objects by generating specialized training samples. Our approach offers a versatile solution that can be seamlessly integrated into various 3D frameworks and 2D semantic segmentation methods, resulting in significantly improved overall detection accuracy. Evaluation on the KITTI and nuScenes datasets demonstrates substantial enhancements in both 3D and birds eye view (BEV) detection benchmarks

{{</citation>}}


### (7/67) SSR-Encoder: Encoding Selective Subject Representation for Subject-Driven Generation (Yuxuan Zhang et al., 2023)

{{<citation>}}

Yuxuan Zhang, Jiaming Liu, Yiren Song, Rui Wang, Hao Tang, Jinpeng Yu, Huaxia Li, Xu Tang, Yao Hu, Han Pan, Zhongliang Jing. (2023)  
**SSR-Encoder: Encoding Selective Subject Representation for Subject-Driven Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.16272v1)  

---


**ABSTRACT**  
Recent advancements in subject-driven image generation have led to zero-shot generation, yet precise selection and focus on crucial subject representations remain challenging. Addressing this, we introduce the SSR-Encoder, a novel architecture designed for selectively capturing any subject from single or multiple reference images. It responds to various query modalities including text and masks, without necessitating test-time fine-tuning. The SSR-Encoder combines a Token-to-Patch Aligner that aligns query inputs with image patches and a Detail-Preserving Subject Encoder for extracting and preserving fine features of the subjects, thereby generating subject embeddings. These embeddings, used in conjunction with original text embeddings, condition the generation process. Characterized by its model generalizability and efficiency, the SSR-Encoder adapts to a range of custom models and control modules. Enhanced by the Embedding Consistency Regularization Loss for improved training, our extensive experiments demonstrate its effectiveness in versatile and high-quality image generation, indicating its broad applicability. Project page: https://ssr-encoder.github.io

{{</citation>}}


### (8/67) Multi-scale Progressive Feature Embedding for Accurate NIR-to-RGB Spectral Domain Translation (Xingxing Yang et al., 2023)

{{<citation>}}

Xingxing Yang, Jie Chen, Zaifeng Yang. (2023)  
**Multi-scale Progressive Feature Embedding for Accurate NIR-to-RGB Spectral Domain Translation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.16040v1)  

---


**ABSTRACT**  
NIR-to-RGB spectral domain translation is a challenging task due to the mapping ambiguities, and existing methods show limited learning capacities. To address these challenges, we propose to colorize NIR images via a multi-scale progressive feature embedding network (MPFNet), with the guidance of grayscale image colorization. Specifically, we first introduce a domain translation module that translates NIR source images into the grayscale target domain. By incorporating a progressive training strategy, the statistical and semantic knowledge from both task domains are efficiently aligned with a series of pixel- and feature-level consistency constraints. Besides, a multi-scale progressive feature embedding network is designed to improve learning capabilities. Experiments show that our MPFNet outperforms state-of-the-art counterparts by 2.55 dB in the NIR-to-RGB spectral domain translation task in terms of PSNR.

{{</citation>}}


### (9/67) Detection-based Intermediate Supervision for Visual Question Answering (Yuhang Liu et al., 2023)

{{<citation>}}

Yuhang Liu, Daowan Peng, Wei Wei, Yuanyuan Fu, Wenfeng Xie, Dangyang Chen. (2023)  
**Detection-based Intermediate Supervision for Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2312.16012v1)  

---


**ABSTRACT**  
Recently, neural module networks (NMNs) have yielded ongoing success in answering compositional visual questions, especially those involving multi-hop visual and logical reasoning. NMNs decompose the complex question into several sub-tasks using instance-modules from the reasoning paths of that question and then exploit intermediate supervisions to guide answer prediction, thereby improving inference interpretability. However, their performance may be hindered due to sketchy modeling of intermediate supervisions. For instance, (1) a prior assumption that each instance-module refers to only one grounded object yet overlooks other potentially associated grounded objects, impeding full cross-modal alignment learning; (2) IoU-based intermediate supervisions may introduce noise signals as the bounding box overlap issue might guide the model's focus towards irrelevant objects. To address these issues, a novel method, \textbf{\underline{D}}etection-based \textbf{\underline{I}}ntermediate \textbf{\underline{S}}upervision (DIS), is proposed, which adopts a generative detection framework to facilitate multiple grounding supervisions via sequence generation. As such, DIS offers more comprehensive and accurate intermediate supervisions, thereby boosting answer prediction performance. Furthermore, by considering intermediate results, DIS enhances the consistency in answering compositional questions and their sub-questions.Extensive experiments demonstrate the superiority of our proposed DIS, showcasing both improved accuracy and state-of-the-art reasoning consistency compared to prior approaches.

{{</citation>}}


### (10/67) Graph Context Transformation Learning for Progressive Correspondence Pruning (Junwen Guo et al., 2023)

{{<citation>}}

Junwen Guo, Guobao Xiao, Shiping Wang, Jun Yu. (2023)  
**Graph Context Transformation Learning for Progressive Correspondence Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15971v1)  

---


**ABSTRACT**  
Most of existing correspondence pruning methods only concentrate on gathering the context information as much as possible while neglecting effective ways to utilize such information. In order to tackle this dilemma, in this paper we propose Graph Context Transformation Network (GCT-Net) enhancing context information to conduct consensus guidance for progressive correspondence pruning. Specifically, we design the Graph Context Enhance Transformer which first generates the graph network and then transforms it into multi-branch graph contexts. Moreover, it employs self-attention and cross-attention to magnify characteristics of each graph context for emphasizing the unique as well as shared essential information. To further apply the recalibrated graph contexts to the global domain, we propose the Graph Context Guidance Transformer. This module adopts a confident-based sampling strategy to temporarily screen high-confidence vertices for guiding accurate classification by searching global consensus between screened vertices and remaining ones. The extensive experimental results on outlier removal and relative pose estimation clearly demonstrate the superior performance of GCT-Net compared to state-of-the-art methods across outdoor and indoor datasets. The source code will be available at: https://github.com/guobaoxiao/GCT-Net/.

{{</citation>}}


### (11/67) Revealing the Proximate Long-Tail Distribution in Compositional Zero-Shot Learning (Chenyi Jiang et al., 2023)

{{<citation>}}

Chenyi Jiang, Haofeng Zhang. (2023)  
**Revealing the Proximate Long-Tail Distribution in Compositional Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.15923v1)  

---


**ABSTRACT**  
Compositional Zero-Shot Learning (CZSL) aims to transfer knowledge from seen state-object pairs to novel unseen pairs. In this process, visual bias caused by the diverse interrelationship of state-object combinations blurs their visual features, hindering the learning of distinguishable class prototypes. Prevailing methods concentrate on disentangling states and objects directly from visual features, disregarding potential enhancements that could arise from a data viewpoint. Experimentally, we unveil the results caused by the above problem closely approximate the long-tailed distribution. As a solution, we transform CZSL into a proximate class imbalance problem. We mathematically deduce the role of class prior within the long-tailed distribution in CZSL. Building upon this insight, we incorporate visual bias caused by compositions into the classifier's training and inference by estimating it as a proximate class prior. This enhancement encourages the classifier to acquire more discernible class prototypes for each composition, thereby achieving more balanced predictions. Experimental results demonstrate that our approach elevates the model's performance to the state-of-the-art level, without introducing additional parameters. Our code is available at \url{https://github.com/LanchJL/ProLT-CZSL}.

{{</citation>}}


### (12/67) ChartBench: A Benchmark for Complex Visual Reasoning in Charts (Zhengzhuo Xu et al., 2023)

{{<citation>}}

Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, Jian Guo. (2023)  
**ChartBench: A Benchmark for Complex Visual Reasoning in Charts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.15915v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have demonstrated remarkable multimodal understanding and generation capabilities. However, their understanding of synthetic charts is limited, while existing benchmarks are simplistic and the charts deviate significantly from real-world examples, making it challenging to accurately assess MLLMs' chart comprehension abilities. Hence, a challenging benchmark is essential for investigating progress and uncovering the limitations of current MLLMs on chart data. In this work, we propose to examine chart comprehension through more complex visual logic and introduce ChartBench, a comprehensive chart benchmark to accurately measure MLLMs' fundamental chart comprehension and data reliability. Specifically, ChartBench consists of \textbf{41} categories, \textbf{2K} charts, and \textbf{16K} QA annotations. While significantly expanding chart types, ChartBench avoids direct labelling of data points, which requires MLLMs to infer values akin to humans by leveraging elements like color, legends, and coordinate systems. We also introduce an improved metric, \textit{Acc+}, which accurately reflects MLLMs' chart comprehension abilities while avoiding labor-intensive manual evaluations or costly GPT-based evaluations. We conduct evaluations on \textbf{12} mainstream open-source models and \textbf{2} outstanding proprietary models. Through extensive experiments, we reveal the limitations of MLLMs on charts and provide insights to inspire the community to pay closer attention to MLLMs' chart comprehension abilities. The benchmark and code will be publicly available for research.

{{</citation>}}


### (13/67) Generating and Reweighting Dense Contrastive Patterns for Unsupervised Anomaly Detection (Songmin Dai et al., 2023)

{{<citation>}}

Songmin Dai, Yifan Wu, Xiaoqiang Li, Xiangyang Xue. (2023)  
**Generating and Reweighting Dense Contrastive Patterns for Unsupervised Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.15911v1)  

---


**ABSTRACT**  
Recent unsupervised anomaly detection methods often rely on feature extractors pretrained with auxiliary datasets or on well-crafted anomaly-simulated samples. However, this might limit their adaptability to an increasing set of anomaly detection tasks due to the priors in the selection of auxiliary datasets or the strategy of anomaly simulation. To tackle this challenge, we first introduce a prior-less anomaly generation paradigm and subsequently develop an innovative unsupervised anomaly detection framework named GRAD, grounded in this paradigm. GRAD comprises three essential components: (1) a diffusion model (PatchDiff) to generate contrastive patterns by preserving the local structures while disregarding the global structures present in normal images, (2) a self-supervised reweighting mechanism to handle the challenge of long-tailed and unlabeled contrastive patterns generated by PatchDiff, and (3) a lightweight patch-level detector to efficiently distinguish the normal patterns and reweighted contrastive patterns. The generation results of PatchDiff effectively expose various types of anomaly patterns, e.g. structural and logical anomaly patterns. In addition, extensive experiments on both MVTec AD and MVTec LOCO datasets also support the aforementioned observation and demonstrate that GRAD achieves competitive anomaly detection accuracy and superior inference speed.

{{</citation>}}


### (14/67) Black-Box Tuning of Vision-Language Models with Effective Gradient Approximation (Zixian Guo et al., 2023)

{{<citation>}}

Zixian Guo, Yuxiang Wei, Ming Liu, Zhilong Ji, Jinfeng Bai, Yiwen Guo, Wangmeng Zuo. (2023)  
**Black-Box Tuning of Vision-Language Models with Effective Gradient Approximation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15901v1)  

---


**ABSTRACT**  
Parameter-efficient fine-tuning (PEFT) methods have provided an effective way for adapting large vision-language models to specific tasks or scenarios. Typically, they learn a very small scale of parameters for pre-trained models in a white-box formulation, which assumes model architectures to be known and parameters to be accessible. However, large models are often not open-source due to considerations of preventing abuse or commercial factors, hence posing a barrier to the deployment of white-box PEFT methods. To alleviate the dependence on model accessibility, we introduce collaborative black-box tuning (CBBT) for both textual prompt optimization and output feature adaptation for black-box models. Specifically, considering that the backpropagation gradients are blocked, we approximate the gradients of textual prompts by analyzing the predictions with perturbed prompts. Secondly, a lightweight adapter is deployed over the output feature of the inaccessible model, further facilitating the model adaptation process. Empowered with these designs, our CBBT is extensively evaluated on eleven downstream benchmarks and achieves remarkable improvements compared to existing black-box VL adaptation methods. Code is released at https://github.com/guozix/cbbt.

{{</citation>}}


### (15/67) Task-Disruptive Background Suppression for Few-Shot Segmentation (Suho Park et al., 2023)

{{<citation>}}

Suho Park, SuBeen Lee, Sangeek Hyun, Hyun Seok Seong, Jae-Pil Heo. (2023)  
**Task-Disruptive Background Suppression for Few-Shot Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.15894v1)  

---


**ABSTRACT**  
Few-shot segmentation aims to accurately segment novel target objects within query images using only a limited number of annotated support images. The recent works exploit support background as well as its foreground to precisely compute the dense correlations between query and support. However, they overlook the characteristics of the background that generally contains various types of objects. In this paper, we highlight this characteristic of background which can bring problematic cases as follows: (1) when the query and support backgrounds are dissimilar and (2) when objects in the support background are similar to the target object in the query. Without any consideration of the above cases, adopting the entire support background leads to a misprediction of the query foreground as background. To address this issue, we propose Task-disruptive Background Suppression (TBS), a module to suppress those disruptive support background features based on two spatial-wise scores: query-relevant and target-relevant scores. The former aims to mitigate the impact of unshared features solely existing in the support background, while the latter aims to reduce the influence of target-similar support background features. Based on these two scores, we define a query background relevant score that captures the similarity between the backgrounds of the query and the support, and utilize it to scale support background features to adaptively restrict the impact of disruptive support backgrounds. Our proposed method achieves state-of-the-art performance on PASCAL-5 and COCO-20 datasets on 1-shot segmentation. Our official code is available at github.com/SuhoPark0706/TBSNet.

{{</citation>}}


### (16/67) Attention-aware Social Graph Transformer Networks for Stochastic Trajectory Prediction (Yao Liu et al., 2023)

{{<citation>}}

Yao Liu, Binghao Li, Xianzhi Wang, Claude Sammut, Lina Yao. (2023)  
**Attention-aware Social Graph Transformer Networks for Stochastic Trajectory Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Graph Convolutional Network, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15881v1)  

---


**ABSTRACT**  
Trajectory prediction is fundamental to various intelligent technologies, such as autonomous driving and robotics. The motion prediction of pedestrians and vehicles helps emergency braking, reduces collisions, and improves traffic safety. Current trajectory prediction research faces problems of complex social interactions, high dynamics and multi-modality. Especially, it still has limitations in long-time prediction. We propose Attention-aware Social Graph Transformer Networks for multi-modal trajectory prediction. We combine Graph Convolutional Networks and Transformer Networks by generating stable resolution pseudo-images from Spatio-temporal graphs through a designed stacking and interception method. Furthermore, we design the attention-aware module to handle social interaction information in scenarios involving mixed pedestrian-vehicle traffic. Thus, we maintain the advantages of the Graph and Transformer, i.e., the ability to aggregate information over an arbitrary number of neighbors and the ability to perform complex time-dependent data processing. We conduct experiments on datasets involving pedestrian, vehicle, and mixed trajectories, respectively. Our results demonstrate that our model minimizes displacement errors across various metrics and significantly reduces the likelihood of collisions. It is worth noting that our model effectively reduces the final displacement error, illustrating the ability of our model to predict for a long time.

{{</citation>}}


### (17/67) SCPMan: Shape Context and Prior Constrained Multi-scale Attention Network for Pancreatic Segmentation (Leilei Zeng et al., 2023)

{{<citation>}}

Leilei Zeng, Xuechen Li, Xinquan Yang, Linlin Shen, Song Wu. (2023)  
**SCPMan: Shape Context and Prior Constrained Multi-scale Attention Network for Pancreatic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2312.15859v1)  

---


**ABSTRACT**  
Due to the poor prognosis of Pancreatic cancer, accurate early detection and segmentation are critical for improving treatment outcomes. However, pancreatic segmentation is challenged by blurred boundaries, high shape variability, and class imbalance. To tackle these problems, we propose a multiscale attention network with shape context and prior constraint for robust pancreas segmentation. Specifically, we proposed a Multi-scale Feature Extraction Module (MFE) and a Mixed-scale Attention Integration Module (MAI) to address unclear pancreas boundaries. Furthermore, a Shape Context Memory (SCM) module is introduced to jointly model semantics across scales and pancreatic shape. Active Shape Model (ASM) is further used to model the shape priors. Experiments on NIH and MSD datasets demonstrate the efficacy of our model, which improves the state-of-the-art Dice Score for 1.01% and 1.03% respectively. Our architecture provides robust segmentation performance, against the blurry boundaries, and variations in scale and shape of pancreas.

{{</citation>}}


### (18/67) Learning Online Policies for Person Tracking in Multi-View Environments (Keivan Nalaie et al., 2023)

{{<citation>}}

Keivan Nalaie, Rong Zheng. (2023)  
**Learning Online Policies for Person Tracking in Multi-View Environments**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15858v1)  

---


**ABSTRACT**  
In this paper, we introduce MVSparse, a novel and efficient framework for cooperative multi-person tracking across multiple synchronized cameras. The MVSparse system is comprised of a carefully orchestrated pipeline, combining edge server-based models with distributed lightweight Reinforcement Learning (RL) agents operating on individual cameras. These RL agents intelligently select informative blocks within each frame based on historical camera data and detection outcomes from neighboring cameras, significantly reducing computational load and communication overhead. The edge server aggregates multiple camera views to perform detection tasks and provides feedback to the individual agents. By projecting inputs from various perspectives onto a common ground plane and applying deep detection models, MVSparse optimally leverages temporal and spatial redundancy in multi-view videos. Notably, our contributions include an empirical analysis of multi-camera pedestrian tracking datasets, the development of a multi-camera, multi-person detection pipeline, and the implementation of MVSparse, yielding impressive results on both open datasets and real-world scenarios. Experimentally, MVSparse accelerates overall inference time by 1.88X and 1.60X compared to a baseline approach while only marginally compromising tracking accuracy by 2.27% and 3.17%, respectively, showcasing its promising potential for efficient multi-camera tracking applications.

{{</citation>}}


### (19/67) Modality-Collaborative Transformer with Hybrid Feature Reconstruction for Robust Emotion Recognition (Chengxin Chen et al., 2023)

{{<citation>}}

Chengxin Chen, Pengyuan Zhang. (2023)  
**Modality-Collaborative Transformer with Hybrid Feature Reconstruction for Robust Emotion Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Emotion Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15848v1)  

---


**ABSTRACT**  
As a vital aspect of affective computing, Multimodal Emotion Recognition has been an active research area in the multimedia community. Despite recent progress, this field still confronts two major challenges in real-world applications: 1) improving the efficiency of constructing joint representations from unaligned multimodal features, and 2) relieving the performance decline caused by random modality feature missing. In this paper, we propose a unified framework, Modality-Collaborative Transformer with Hybrid Feature Reconstruction (MCT-HFR), to address these issues. The crucial component of MCT is a novel attention-based encoder which concurrently extracts and dynamically balances the intra- and inter-modality relations for all associated modalities. With additional modality-wise parameter sharing, a more compact representation can be encoded with less time and space complexity. To improve the robustness of MCT, we further introduce HFR which consists of two modules: Local Feature Imagination (LFI) and Global Feature Alignment (GFA). During model training, LFI leverages complete features as supervisory signals to recover local missing features, while GFA is designed to reduce the global semantic gap between pairwise complete and incomplete representations. Experimental evaluations on two popular benchmark datasets demonstrate that our proposed method consistently outperforms advanced baselines in both complete and incomplete data scenarios.

{{</citation>}}


## cs.CL (21)



### (20/67) Task Contamination: Language Models May Not Be Few-Shot Anymore (Changmao Li et al., 2023)

{{<citation>}}

Changmao Li, Jeffrey Flanigan. (2023)  
**Task Contamination: Language Models May Not Be Few-Shot Anymore**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Few-Shot, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16337v1)  

---


**ABSTRACT**  
Large language models (LLMs) offer impressive performance in various zero-shot and few-shot tasks. However, their success in zero-shot and few-shot settings may be affected by task contamination, a potential limitation that has not been thoroughly examined. This paper investigates how zero-shot and few-shot performance of LLMs has changed chronologically over time. Utilizing GPT-3 series models and several other recent open-sourced LLMs, and controlling for dataset difficulty, we find that on datasets released before the LLM training data creation date, LLMs perform surprisingly better than on datasets released after. This strongly indicates that, for many LLMs, there exists task contamination on zero-shot and few-shot evaluation for datasets released prior to the LLMs' training data creation date. Additionally, we utilize training data inspection, task example extraction, and a membership inference attack, which reveal further evidence of task contamination. Importantly, we find that for classification tasks with no possibility of task contamination, LLMs rarely demonstrate statistically significant improvements over simple majority baselines, in both zero and few-shot settings.

{{</citation>}}


### (21/67) Zur Darstellung eines mehrstufigen Prototypbegriffs in der multilingualen automatischen Sprachgenerierung: vom Korpus über word embeddings bis hin zum automatischen Wörterbuch (María José Domínguez Vázquez, 2023)

{{<citation>}}

María José Domínguez Vázquez. (2023)  
**Zur Darstellung eines mehrstufigen Prototypbegriffs in der multilingualen automatischen Sprachgenerierung: vom Korpus über word embeddings bis hin zum automatischen Wörterbuch**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.16311v1)  

---


**ABSTRACT**  
The multilingual dictionary of noun valency Portlex is considered to be the trigger for the creation of the automatic language generators Xera and Combinatoria, whose development and use is presented in this paper. Both prototypes are used for the automatic generation of nominal phrases with their mono- and bi-argumental valence slots, which could be used, among others, as dictionary examples or as integrated components of future autonomous E-Learning-Tools. As samples for new types of automatic valency dictionaries including user interaction, we consider the language generators as we know them today. In the specific methodological procedure for the development of the language generators, the syntactic-semantic description of the noun slots turns out to be the main focus from a syntagmatic and paradigmatic point of view. Along with factors such as representativeness, grammatical correctness, semantic coherence, frequency and the variety of lexical candidates, as well as semantic classes and argument structures, which are fixed components of both resources, a concept of a multi-sided prototype stands out. The combined application of this prototype concept as well as of word embeddings together with techniques from the field of automatic natural language processing and generation (NLP and NLG) opens up a new way for the future development of automatically generated plurilingual valency dictionaries. All things considered, the paper depicts the language generators both from the point of view of their development as well as from that of the users. The focus lies on the role of the prototype concept within the development of the resources.

{{</citation>}}


### (22/67) Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4 (Sondos Mahmoud Bsharat et al., 2023)

{{<citation>}}

Sondos Mahmoud Bsharat, Aidar Myrzakhan, Zhiqiang Shen. (2023)  
**Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA  
[Paper Link](http://arxiv.org/abs/2312.16171v1)  

---


**ABSTRACT**  
This paper introduces 26 guiding principles designed to streamline the process of querying and prompting large language models. Our goal is to simplify the underlying concepts of formulating questions for various scales of large language models, examining their abilities, and enhancing user comprehension on the behaviors of different scales of large language models when feeding into different prompts. Extensive experiments are conducted on LLaMA-1/2 (7B, 13B and 70B), GPT-3.5/4 to verify the effectiveness of the proposed principles on instructions and prompts design. We hope that this work provides a better guide for researchers working on the prompting of large language models. Project page is available at https://github.com/VILA-Lab/ATLAS.

{{</citation>}}


### (23/67) From Text to Multimodal: A Comprehensive Survey of Adversarial Example Generation in Question Answering Systems (Gulsum Yigit et al., 2023)

{{<citation>}}

Gulsum Yigit, Mehmet Fatih Amasyali. (2023)  
**From Text to Multimodal: A Comprehensive Survey of Adversarial Example Generation in Question Answering Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.16156v1)  

---


**ABSTRACT**  
Integrating adversarial machine learning with Question Answering (QA) systems has emerged as a critical area for understanding the vulnerabilities and robustness of these systems. This article aims to comprehensively review adversarial example-generation techniques in the QA field, including textual and multimodal contexts. We examine the techniques employed through systematic categorization, providing a comprehensive, structured review. Beginning with an overview of traditional QA models, we traverse the adversarial example generation by exploring rule-based perturbations and advanced generative models. We then extend our research to include multimodal QA systems, analyze them across various methods, and examine generative models, seq2seq architectures, and hybrid methodologies. Our research grows to different defense strategies, adversarial datasets, and evaluation metrics and illustrates the comprehensive literature on adversarial QA. Finally, the paper considers the future landscape of adversarial question generation, highlighting potential research directions that can advance textual and multimodal QA systems in the context of adversarial challenges.

{{</citation>}}


### (24/67) The Media Bias Taxonomy: A Systematic Literature Review on the Forms and Automated Detection of Media Bias (Timo Spinde et al., 2023)

{{<citation>}}

Timo Spinde, Smilla Hinterreiter, Fabian Haak, Terry Ruas, Helge Giese, Norman Meuschke, Bela Gipp. (2023)  
**The Media Bias Taxonomy: A Systematic Literature Review on the Forms and Automated Detection of Media Bias**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.16148v2)  

---


**ABSTRACT**  
The way the media presents events can significantly affect public perception, which in turn can alter people's beliefs and views. Media bias describes a one-sided or polarizing perspective on a topic. This article summarizes the research on computational methods to detect media bias by systematically reviewing 3140 research papers published between 2019 and 2022. To structure our review and support a mutual understanding of bias across research domains, we introduce the Media Bias Taxonomy, which provides a coherent overview of the current state of research on media bias from different perspectives. We show that media bias detection is a highly active research field, in which transformer-based classification approaches have led to significant improvements in recent years. These improvements include higher classification accuracy and the ability to detect more fine-granular types of bias. However, we have identified a lack of interdisciplinarity in existing projects, and a need for more awareness of the various types of media bias to support methodologically thorough performance evaluations of media bias detection systems. Concluding from our analysis, we see the integration of recent machine learning advancements with reliable and diverse bias assessment strategies from other research areas as the most promising area for future research contributions in the field.

{{</citation>}}


### (25/67) JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings for Retrieval: Early Technical Report (Benjamin Clavié, 2023)

{{<citation>}}

Benjamin Clavié. (2023)  
**JaColBERT and Hard Negatives, Towards Better Japanese-First Embeddings for Retrieval: Early Technical Report**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Embedding  
[Paper Link](http://arxiv.org/abs/2312.16144v1)  

---


**ABSTRACT**  
Document retrieval in many languages has been largely relying on multi-lingual models, and leveraging the vast wealth of English training data. In Japanese, the best performing deep-learning based retrieval approaches rely on multilingual dense embeddings. In this work, we introduce (1) a hard-negative augmented version of the Japanese MMARCO dataset and (2) JaColBERT, a document retrieval model built on the ColBERT model architecture, specifically for Japanese. JaColBERT vastly outperform all previous monolingual retrieval approaches and competes with the best multilingual methods, despite unfavourable evaluation settings (out-of-domain vs. in-domain for the multilingual models). JaColBERT reaches an average Recall@10 of 0.813, noticeably ahead of the previous monolingual best-performing model (0.716) and only slightly behind multilingual-e5-base (0.820), though more noticeably behind multilingual-e5-large (0.856). These results are achieved using only a limited, entirely Japanese, training set, more than two orders of magnitudes smaller than multilingual embedding models. We believe these results show great promise to support retrieval-enhanced application pipelines in a wide variety of domains.

{{</citation>}}


### (26/67) RoleEval: A Bilingual Role Evaluation Benchmark for Large Language Models (Tianhao Shen et al., 2023)

{{<citation>}}

Tianhao Shen, Sun Li, Deyi Xiong. (2023)  
**RoleEval: A Bilingual Role Evaluation Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16132v1)  

---


**ABSTRACT**  
The rapid evolution of large language models (LLMs) necessitates effective benchmarks for evaluating their role knowledge, which is essential for establishing connections with the real world and providing more immersive interactions. This paper introduces RoleEval, a bilingual benchmark designed to assess the memorization, utilization, and reasoning capabilities of role knowledge. RoleEval comprises RoleEval-Global (including internationally recognized characters) and RoleEval-Chinese (including characters popular in China), with 6,000 Chinese-English parallel multiple-choice questions focusing on 300 influential people and fictional characters drawn from a variety of domains including celebrities, anime, comics, movies, TV series, games, and fiction. These questions cover basic knowledge and multi-hop reasoning abilities, aiming to systematically probe various aspects such as personal information, relationships, abilities, and experiences of the characters. To maintain high standards, we perform a hybrid quality check process combining automatic and human verification, ensuring that the questions are diverse, challenging, and discriminative.   Our extensive evaluations of RoleEval across various open-source and proprietary large language models, under both the zero- and few-shot settings, reveal insightful findings. Notably, while GPT-4 outperforms other models on RoleEval-Global, Chinese LLMs excel on RoleEval-Chinese, highlighting significant knowledge distribution differences. We expect that RoleEval will highlight the significance of assessing role knowledge for foundation models across various languages and cultural settings.

{{</citation>}}


### (27/67) Dotless Representation of Arabic Text: Analysis and Modeling (Maged S. Al-Shaibani et al., 2023)

{{<citation>}}

Maged S. Al-Shaibani, Irfan Ahmad. (2023)  
**Dotless Representation of Arabic Text: Analysis and Modeling**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.16104v1)  

---


**ABSTRACT**  
This paper presents a novel dotless representation of Arabic text as an alternative to the standard Arabic text representation. We delve into its implications through comprehensive analysis across five diverse corpora and four different tokenization techniques. We explore the impact of dotless representation on the relationships between tokenization granularity and vocabulary size and compare them with standard text representation. Moreover, we analyze the information density of dotless versus standard text using text entropy calculations. To delve deeper into the implications of the dotless representation, statistical and neural language models are constructed using the various text corpora and tokenization techniques. A comparative assessment is then made against language models developed using the standard Arabic text representation. This multifaceted analysis provides valuable insights into the potential advantages and challenges associated with the dotless representation. Last but not the least, utilizing parallel corpora, we draw comparisons between the text analysis of Arabic and English to gain further insights. Our findings shed light on the potential benefits of dotless representation for various NLP tasks, paving the way for further exploration for Arabic natural language processing.

{{</citation>}}


### (28/67) A Logically Consistent Chain-of-Thought Approach for Stance Detection (Bowen Zhang et al., 2023)

{{<citation>}}

Bowen Zhang, Daijun Ding, Liwen Jing, Hu Huang. (2023)  
**A Logically Consistent Chain-of-Thought Approach for Stance Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Stance Detection  
[Paper Link](http://arxiv.org/abs/2312.16054v1)  

---


**ABSTRACT**  
Zero-shot stance detection (ZSSD) aims to detect stances toward unseen targets. Incorporating background knowledge to enhance transferability between seen and unseen targets constitutes the primary approach of ZSSD. However, these methods often struggle with a knowledge-task disconnect and lack logical consistency in their predictions. To address these issues, we introduce a novel approach named Logically Consistent Chain-of-Thought (LC-CoT) for ZSSD, which improves stance detection by ensuring relevant and logically sound knowledge extraction. LC-CoT employs a three-step process. Initially, it assesses whether supplementary external knowledge is necessary. Subsequently, it uses API calls to retrieve this knowledge, which can be processed by a separate LLM. Finally, a manual exemplar guides the LLM to infer stance categories, using an if-then logical structure to maintain relevance and logical coherence. This structured approach to eliciting background knowledge enhances the model's capability, outperforming traditional supervised methods without relying on labeled data.

{{</citation>}}


### (29/67) Aligning Large Language Models with Human Preferences through Representation Engineering (Wenhao Liu et al., 2023)

{{<citation>}}

Wenhao Liu, Xiaohua Wang, Muling Wu, Tianlong Li, Changze Lv, Zixuan Ling, Jianhao Zhu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang. (2023)  
**Aligning Large Language Models with Human Preferences through Representation Engineering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15997v1)  

---


**ABSTRACT**  
Aligning large language models (LLMs) with human preferences is crucial for enhancing their utility in terms of helpfulness, truthfulness, safety, harmlessness, and interestingness. Existing methods for achieving this alignment often involves employing reinforcement learning from human feedback (RLHF) to fine-tune LLMs based on human labels assessing the relative quality of model responses. Nevertheless, RLHF is susceptible to instability during fine-tuning and presents challenges in implementation.Drawing inspiration from the emerging field of representation engineering (RepE), this study aims to identify relevant representations for high-level human preferences embedded in patterns of activity within an LLM, and achieve precise control of model behavior by transforming its representations. This novel approach, denoted as Representation Alignment from Human Feedback (RAHF), proves to be effective, computationally efficient, and easy to implement.Extensive experiments demonstrate the efficacy of RAHF in not only capturing but also manipulating representations to align with a broad spectrum of human preferences or values, rather than being confined to a singular concept or function (e.g. honesty or bias). RAHF's versatility in accommodating diverse human preferences shows its potential for advancing LLM performance.

{{</citation>}}


### (30/67) Towards Probing Contact Center Large Language Models (Varun Nathan et al., 2023)

{{<citation>}}

Varun Nathan, Ayush Kumar, Digvijay Ingle, Jithendra Vepa. (2023)  
**Towards Probing Contact Center Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, T5  
[Paper Link](http://arxiv.org/abs/2312.15922v1)  

---


**ABSTRACT**  
Fine-tuning large language models (LLMs) with domain-specific instructions has emerged as an effective method to enhance their domain-specific understanding. Yet, there is limited work that examines the core characteristics acquired during this process. In this study, we benchmark the fundamental characteristics learned by contact-center (CC) specific instruction fine-tuned LLMs with out-of-the-box (OOB) LLMs via probing tasks encompassing conversational, channel, and automatic speech recognition (ASR) properties. We explore different LLM architectures (Flan-T5 and Llama), sizes (3B, 7B, 11B, 13B), and fine-tuning paradigms (full fine-tuning vs PEFT). Our findings reveal remarkable effectiveness of CC-LLMs on the in-domain downstream tasks, with improvement in response acceptability by over 48% compared to OOB-LLMs. Additionally, we compare the performance of OOB-LLMs and CC-LLMs on the widely used SentEval dataset, and assess their capabilities in terms of surface, syntactic, and semantic information through probing tasks. Intriguingly, we note a relatively consistent performance of probing classifiers on the set of probing tasks. Our observations indicate that CC-LLMs, while outperforming their out-of-the-box counterparts, exhibit a tendency to rely less on encoding surface, syntactic, and semantic properties, highlighting the intricate interplay between domain-specific adaptation and probing task performance opening up opportunities to explore behavior of fine-tuned language models in specialized contexts.

{{</citation>}}


### (31/67) Supervised Knowledge Makes Large Language Models Better In-context Learners (Linyi Yang et al., 2023)

{{<citation>}}

Linyi Yang, Shuibai Zhang, Zhuohao Yu, Guangsheng Bao, Yidong Wang, Jindong Wang, Ruochen Xu, Wei Ye, Xing Xie, Weizhu Chen, Yue Zhang. (2023)  
**Supervised Knowledge Makes Large Language Models Better In-context Learners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15918v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) exhibit emerging in-context learning abilities through prompt engineering. The recent progress in large-scale generative models has further expanded their use in real-world language applications. However, the critical challenge of improving the generalizability and factuality of LLMs in natural language understanding and question answering remains under-explored. While previous in-context learning research has focused on enhancing models to adhere to users' specific instructions and quality expectations, and to avoid undesired outputs, little to no work has explored the use of task-Specific fine-tuned Language Models (SLMs) to improve LLMs' in-context learning during the inference stage. Our primary contribution is the establishment of a simple yet effective framework that enhances the reliability of LLMs as it: 1) generalizes out-of-distribution data, 2) elucidates how LLMs benefit from discriminative models, and 3) minimizes hallucinations in generative tasks. Using our proposed plug-in method, enhanced versions of Llama 2 and ChatGPT surpass their original versions regarding generalizability and factuality. We offer a comprehensive suite of resources, including 16 curated datasets, prompts, model checkpoints, and LLM outputs across 9 distinct tasks. Our empirical analysis sheds light on the advantages of incorporating discriminative models into LLMs and highlights the potential of our methodology in fostering more reliable LLMs.

{{</citation>}}


### (32/67) Align on the Fly: Adapting Chatbot Behavior to Established Norms (Chunpu Xu et al., 2023)

{{<citation>}}

Chunpu Xu, Steffi Chern, Ethan Chern, Ge Zhang, Zekun Wang, Ruibo Liu, Jing Li, Jie Fu, Pengfei Liu. (2023)  
**Align on the Fly: Adapting Chatbot Behavior to Established Norms**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2312.15907v1)  

---


**ABSTRACT**  
In this paper, we aim to align large language models with the ever-changing, complex, and diverse human values (e.g., social norms) across time and locations. This presents a challenge to existing alignment techniques, such as supervised fine-tuning, which internalize values within model parameters. To overcome this, we propose an On-the-fly Preference Optimization (OPO) method, which is a real-time alignment that works in a streaming way. It employs an external memory to store established rules for alignment, which can constrain LLMs' behaviors without further training, allowing for convenient updates and customization of human values. We also introduce a scalable evaluation to assess the proposed method more effectively. Experimental results on both human-annotated and auto-generated questions from legal and moral domains indicate the effectiveness of the proposed OPO method. Our code and data are released at https://github.com/GAIR-NLP/OPO.

{{</citation>}}


### (33/67) Think and Retrieval: A Hypothesis Knowledge Graph Enhanced Medical Large Language Models (Xinke Jiang et al., 2023)

{{<citation>}}

Xinke Jiang, Ruizhe Zhang, Yongxin Xu, Rihong Qiu, Yue Fang, Zhiyuan Wang, Jinyi Tang, Hongxin Ding, Xu Chu, Junfeng Zhao, Yasha Wang. (2023)  
**Think and Retrieval: A Hypothesis Knowledge Graph Enhanced Medical Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.15883v1)  

---


**ABSTRACT**  
We explore how the rise of Large Language Models (LLMs) significantly impacts task performance in the field of Natural Language Processing. We focus on two strategies, Retrieval-Augmented Generation (RAG) and Fine-Tuning (FT), and propose the Hypothesis Knowledge Graph Enhanced (HyKGE) framework, leveraging a knowledge graph to enhance medical LLMs. By integrating LLMs and knowledge graphs, HyKGE demonstrates superior performance in addressing accuracy and interpretability challenges, presenting potential applications in the medical domain. Our evaluations using real-world datasets highlight HyKGE's superiority in providing accurate knowledge with precise confidence, particularly in complex and difficult scenarios. The code will be available until published.

{{</citation>}}


### (34/67) KnowledgeNavigator: Leveraging Large Language Models for Enhanced Reasoning over Knowledge Graph (Tiezheng Guo et al., 2023)

{{<citation>}}

Tiezheng Guo, Qingwen Yang, Chen Wang, Yanyi Liu, Pan Li, Jiawei Tang, Dapeng Li, Yingyou Wen. (2023)  
**KnowledgeNavigator: Leveraging Large Language Models for Enhanced Reasoning over Knowledge Graph**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.15880v1)  

---


**ABSTRACT**  
Large language model (LLM) has achieved outstanding performance on various downstream tasks with its powerful natural language understanding and zero-shot capability, but LLM still suffers from knowledge limitation. Especially in scenarios that require long logical chains or complex reasoning, the hallucination and knowledge limitation of LLM limit its performance in question answering (QA). In this paper, we propose a novel framework KnowledgeNavigator to address these challenges by efficiently and accurately retrieving external knowledge from knowledge graph and using it as a key factor to enhance LLM reasoning. Specifically, KnowledgeNavigator first mines and enhances the potential constraints of the given question to guide the reasoning. Then it retrieves and filters external knowledge that supports answering through iterative reasoning on knowledge graph with the guidance of LLM and the question. Finally, KnowledgeNavigator constructs the structured knowledge into effective prompts that are friendly to LLM to help its reasoning. We evaluate KnowledgeNavigator on multiple public KGQA benchmarks, the experiments show the framework has great effectiveness and generalization, outperforming previous knowledge graph enhanced LLM methods and is comparable to the fully supervised models.

{{</citation>}}


### (35/67) Heterogeneous Encoders Scaling In The Transformer For Neural Machine Translation (Jia Cheng Hu et al., 2023)

{{<citation>}}

Jia Cheng Hu, Roberto Cavicchioli, Giulia Berardinelli, Alessandro Capotondi. (2023)  
**Heterogeneous Encoders Scaling In The Transformer For Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15872v1)  

---


**ABSTRACT**  
Although the Transformer is currently the best-performing architecture in the homogeneous configuration (self-attention only) in Neural Machine Translation, many State-of-the-Art models in Natural Language Processing are made of a combination of different Deep Learning approaches. However, these models often focus on combining a couple of techniques only and it is unclear why some methods are chosen over others. In this work, we investigate the effectiveness of integrating an increasing number of heterogeneous methods. Based on a simple combination strategy and performance-driven synergy criteria, we designed the Multi-Encoder Transformer, which consists of up to five diverse encoders. Results showcased that our approach can improve the quality of the translation across a variety of languages and dataset sizes and it is particularly effective in low-resource languages where we observed a maximum increase of 7.16 BLEU compared to the single-encoder model.

{{</citation>}}


### (36/67) Medical Report Generation based on Segment-Enhanced Contrastive Representation Learning (Ruoqing Zhao et al., 2023)

{{<citation>}}

Ruoqing Zhao, Xi Wang, Hongliang Dai, Pan Gao, Piji Li. (2023)  
**Medical Report Generation based on Segment-Enhanced Contrastive Representation Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Contrastive Learning, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.15869v1)  

---


**ABSTRACT**  
Automated radiology report generation has the potential to improve radiology reporting and alleviate the workload of radiologists. However, the medical report generation task poses unique challenges due to the limited availability of medical data and the presence of data bias. To maximize the utility of available data and reduce data bias, we propose MSCL (Medical image Segmentation with Contrastive Learning), a framework that utilizes the Segment Anything Model (SAM) to segment organs, abnormalities, bones, etc., and can pay more attention to the meaningful ROIs in the image to get better visual representations. Then we introduce a supervised contrastive loss that assigns more weight to reports that are semantically similar to the target while training. The design of this loss function aims to mitigate the impact of data bias and encourage the model to capture the essential features of a medical image and generate high-quality reports. Experimental results demonstrate the effectiveness of our proposed model, where we achieve state-of-the-art performance on the IU X-Ray public dataset.

{{</citation>}}


### (37/67) Punctuation Matters! Stealthy Backdoor Attack for Language Models (Xuan Sheng et al., 2023)

{{<citation>}}

Xuan Sheng, Zhicheng Li, Zhaoyang Han, Xiangmao Chang, Piji Li. (2023)  
**Punctuation Matters! Stealthy Backdoor Attack for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.15867v1)  

---


**ABSTRACT**  
Recent studies have pointed out that natural language processing (NLP) models are vulnerable to backdoor attacks. A backdoored model produces normal outputs on the clean samples while performing improperly on the texts with triggers that the adversary injects. However, previous studies on textual backdoor attack pay little attention to stealthiness. Moreover, some attack methods even cause grammatical issues or change the semantic meaning of the original texts. Therefore, they can easily be detected by humans or defense systems. In this paper, we propose a novel stealthy backdoor attack method against textual models, which is called \textbf{PuncAttack}. It leverages combinations of punctuation marks as the trigger and chooses proper locations strategically to replace them. Through extensive experiments, we demonstrate that the proposed method can effectively compromise multiple models in various tasks. Meanwhile, we conduct automatic evaluation and human inspection, which indicate the proposed method possesses good performance of stealthiness without bringing grammatical issues and altering the meaning of sentences.

{{</citation>}}


### (38/67) More than Correlation: Do Large Language Models Learn Causal Representations of Space? (Yida Chen et al., 2023)

{{<citation>}}

Yida Chen, Yixian Gan, Sijia Li, Li Yao, Xiaohan Zhao. (2023)  
**More than Correlation: Do Large Language Models Learn Causal Representations of Space?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16257v1)  

---


**ABSTRACT**  
Recent work found high mutual information between the learned representations of large language models (LLMs) and the geospatial property of its input, hinting an emergent internal model of space. However, whether this internal space model has any causal effects on the LLMs' behaviors was not answered by that work, led to criticism of these findings as mere statistical correlation. Our study focused on uncovering the causality of the spatial representations in LLMs. In particular, we discovered the potential spatial representations in DeBERTa, GPT-Neo using representational similarity analysis and linear and non-linear probing. Our casual intervention experiments showed that the spatial representations influenced the model's performance on next word prediction and a downstream task that relies on geospatial information. Our experiments suggested that the LLMs learn and use an internal model of space in solving geospatial related tasks.

{{</citation>}}


### (39/67) Knowledge Distillation of LLM for Education (Ehsan Latif et al., 2023)

{{<citation>}}

Ehsan Latif, Luyang Fang, Ping Ma, Xiaoming Zhai. (2023)  
**Knowledge Distillation of LLM for Education**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Knowledge Distillation, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15842v1)  

---


**ABSTRACT**  
This study proposes a method for distilling the knowledge of fine-tuned Large Language Models (LLMs) into a smaller, more efficient, and accurate neural network, specifically targeting the challenge of deploying these models on resource-constrained devices. Our methodology involves training the smaller student model using the prediction probabilities of the LLM, which serves as a teacher model. This is achieved through a specialized loss function tailored to learn from the LLM's output probabilities, ensuring that the student model closely mimics the teacher's performance. To test this approach, we utilized a large dataset, 7T, containing 6,684 student-written responses to science questions and three other datasets with student-written responses. We also compared performance with original neural network (NN) models to validate the accuracy. Results have shown that the NN and distilled student models have comparable accuracy to the teacher model for the 7T dataset; however, other datasets have shown significantly lower accuracy (28% on average) for NN, though our proposed distilled model is still able to achieve 12\% higher accuracy than NN. Furthermore, the student model size ranges from 0.1M to 0.02M, 100 times smaller in terms of parameters and ten times smaller compared with the original output model size. The significance of this research lies in its potential to make advanced AI technologies accessible in typical educational settings, particularly for automatic scoring.

{{</citation>}}


### (40/67) SecQA: A Concise Question-Answering Dataset for Evaluating Large Language Models in Computer Security (Zefang Liu, 2023)

{{<citation>}}

Zefang Liu. (2023)  
**SecQA: A Concise Question-Answering Dataset for Evaluating Large Language Models in Computer Security**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, QA, Security  
[Paper Link](http://arxiv.org/abs/2312.15838v1)  

---


**ABSTRACT**  
In this paper, we introduce SecQA, a novel dataset tailored for evaluating the performance of Large Language Models (LLMs) in the domain of computer security. Utilizing multiple-choice questions generated by GPT-4 based on the "Computer Systems Security: Planning for Success" textbook, SecQA aims to assess LLMs' understanding and application of security principles. We detail the structure and intent of SecQA, which includes two versions of increasing complexity, to provide a concise evaluation across various difficulty levels. Additionally, we present an extensive evaluation of prominent LLMs, including GPT-3.5-Turbo, GPT-4, Llama-2, Vicuna, Mistral, and Zephyr models, using both 0-shot and 5-shot learning settings. Our results, encapsulated in the SecQA v1 and v2 datasets, highlight the varying capabilities and limitations of these models in the computer security context. This study not only offers insights into the current state of LLMs in understanding security-related content but also establishes SecQA as a benchmark for future advancements in this critical research area.

{{</citation>}}


## cs.LG (13)



### (41/67) Observable Propagation: A Data-Efficient Approach to Uncover Feature Vectors in Transformers (Jacob Dunefsky et al., 2023)

{{<citation>}}

Jacob Dunefsky, Arman Cohan. (2023)  
**Observable Propagation: A Data-Efficient Approach to Uncover Feature Vectors in Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.16291v1)  

---


**ABSTRACT**  
A key goal of current mechanistic interpretability research in NLP is to find linear features (also called "feature vectors") for transformers: directions in activation space corresponding to concepts that are used by a given model in its computation. Present state-of-the-art methods for finding linear features require large amounts of labelled data -- both laborious to acquire and computationally expensive to utilize. In this work, we introduce a novel method, called "observable propagation" (in short: ObsProp), for finding linear features used by transformer language models in computing a given task -- using almost no data. Our paradigm centers on the concept of observables, linear functionals corresponding to given tasks. We then introduce a mathematical theory for the analysis of feature vectors: we provide theoretical motivation for why LayerNorm nonlinearities do not affect the direction of feature vectors; we also introduce a similarity metric between feature vectors called the coupling coefficient which estimates the degree to which one feature's output correlates with another's. We use ObsProp to perform extensive qualitative investigations into several tasks, including gendered occupational bias, political party prediction, and programming language detection. Our results suggest that ObsProp surpasses traditional approaches for finding feature vectors in the low-data regime, and that ObsProp can be used to better understand the mechanisms responsible for bias in large language models. Code for experiments can be found at github.com/jacobdunefsky/ObservablePropagation.

{{</citation>}}


### (42/67) A bi-objective $ε$-constrained framework for quality-cost optimization in language model ensembles (Aditi Singla et al., 2023)

{{<citation>}}

Aditi Singla, Aditya Singh, Kanishk Kukreja. (2023)  
**A bi-objective $ε$-constrained framework for quality-cost optimization in language model ensembles**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-NE, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16119v1)  

---


**ABSTRACT**  
We propose an ensembling framework that uses diverse open-sourced Large Language Models (LLMs) to achieve high response quality while maintaining cost efficiency. We formulate a bi-objective optimization problem to represent the quality-cost tradeoff and then introduce an additional budget constraint that reduces the problem to a straightforward 0/1 knapsack problem. We empirically demonstrate that our framework outperforms the existing ensembling approaches in response quality while significantly reducing costs.

{{</citation>}}


### (43/67) Algebraic Positional Encodings (Konstantinos Kogkalidis et al., 2023)

{{<citation>}}

Konstantinos Kogkalidis, Jean-Philippe Bernardy, Vikas Garg. (2023)  
**Algebraic Positional Encodings**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.16045v1)  

---


**ABSTRACT**  
We introduce a novel positional encoding strategy for Transformer-style models, addressing the shortcomings of existing, often ad hoc, approaches. Our framework provides a flexible mapping from the algebraic specification of a domain to an interpretation as orthogonal operators. This design preserves the algebraic characteristics of the source domain, ensuring that the model upholds the desired structural properties. Our scheme can accommodate various structures, including sequences, grids and trees, as well as their compositions. We conduct a series of experiments to demonstrate the practical applicability of our approach. Results suggest performance on par with or surpassing the current state-of-the-art, without hyperparameter optimizations or ``task search'' of any kind. Code will be made available at \url{github.com/konstantinosKokos/UnitaryPE}.

{{</citation>}}


### (44/67) Robust Neural Pruning with Gradient Sampling Optimization for Residual Neural Networks (Juyoung Yun, 2023)

{{<citation>}}

Juyoung Yun. (2023)  
**Robust Neural Pruning with Gradient Sampling Optimization for Residual Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.16020v1)  

---


**ABSTRACT**  
In this study, we explore an innovative approach for neural network optimization, focusing on the application of gradient sampling techniques, similar to those in StochGradAdam, during the pruning process. Our primary objective is to maintain high accuracy levels in pruned models, a critical challenge in resource-limited scenarios. Our extensive experiments reveal that models optimized with gradient sampling techniques are more effective at preserving accuracy during pruning compared to those using traditional optimization methods. This finding underscores the significance of gradient sampling in facilitating robust learning and enabling networks to retain crucial information even after substantial reduction in their complexity. We validate our approach across various datasets and neural architectures, demonstrating its broad applicability and effectiveness. The paper also delves into the theoretical aspects, explaining how gradient sampling techniques contribute to the robustness of models during pruning. Our results suggest a promising direction for creating efficient neural networks that do not compromise on accuracy, even in environments with constrained computational resources.

{{</citation>}}


### (45/67) Practical Bias Mitigation through Proxy Sensitive Attribute Label Generation (Bhushan Chaudhary et al., 2023)

{{<citation>}}

Bhushan Chaudhary, Anubha Pandey, Deepak Bhatt, Darshika Tiwari. (2023)  
**Practical Bias Mitigation through Proxy Sensitive Attribute Label Generation**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.15994v1)  

---


**ABSTRACT**  
Addressing bias in the trained machine learning system often requires access to sensitive attributes. In practice, these attributes are not available either due to legal and policy regulations or data unavailability for a given demographic. Existing bias mitigation algorithms are limited in their applicability to real-world scenarios as they require access to sensitive attributes to achieve fairness. In this research work, we aim to address this bottleneck through our proposed unsupervised proxy-sensitive attribute label generation technique. Towards this end, we propose a two-stage approach of unsupervised embedding generation followed by clustering to obtain proxy-sensitive labels. The efficacy of our work relies on the assumption that bias propagates through non-sensitive attributes that are correlated to the sensitive attributes and, when mapped to the high dimensional latent space, produces clusters of different demographic groups that exist in the data. Experimental results demonstrate that bias mitigation using existing algorithms such as Fair Mixup and Adversarial Debiasing yields comparable results on derived proxy labels when compared against using true sensitive attributes.

{{</citation>}}


### (46/67) Optimistic and Pessimistic Actor in RL:Decoupling Exploration and Utilization (Jingpu Yang et al., 2023)

{{<citation>}}

Jingpu Yang, Qirui Zhao, Helin Wang, Yuxiao Huang, Zirui Song, Miao Fang. (2023)  
**Optimistic and Pessimistic Actor in RL:Decoupling Exploration and Utilization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15965v1)  

---


**ABSTRACT**  
Deep neural network(DNN) generalization is limited by the over-reliance of current offline reinforcement learning techniques on conservative processing of existing datasets. This method frequently results in algorithms that settle for suboptimal solutions that only adjust to a certain dataset. Similarly, in online reinforcement learning, the previously imposed punitive pessimism also deprives the model of its exploratory potential. Our research proposes a novel framework, Optimistic and Pessimistic Actor Reinforcement Learning (OPARL). OPARL employs a unique dual-actor approach: an optimistic actor dedicated to exploration and a pessimistic actor focused on utilization, thereby effectively differentiating between exploration and utilization strategies. This unique combination in reinforcement learning methods fosters a more balanced and efficient approach. It enables the optimization of policies that focus on actions yielding high rewards through pessimistic utilization strategies, while also ensuring extensive state coverage via optimistic exploration. Experiments and theoretical study demonstrates OPARL improves agents' capacities for application and exploration. In the most tasks of DMControl benchmark and Mujoco environment, OPARL performed better than state-of-the-art methods. Our code has released on https://github.com/yydsok/OPARL

{{</citation>}}


### (47/67) MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks (Jingyao Li et al., 2023)

{{<citation>}}

Jingyao Li, Pengguang Chen, Jiaya Jia. (2023)  
**MoTCoder: Elevating Large Language Models with Modular of Thought for Challenging Programming Tasks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-PL, cs-SE, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15960v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have showcased impressive capabilities in handling straightforward programming tasks. However, their performance tends to falter when confronted with more challenging programming problems. We observe that conventional models often generate solutions as monolithic code blocks, restricting their effectiveness in tackling intricate questions. To overcome this limitation, we present Modular-of-Thought Coder (MoTCoder). We introduce a pioneering framework for MoT instruction tuning, designed to promote the decomposition of tasks into logical sub-tasks and sub-modules. Our investigations reveal that, through the cultivation and utilization of sub-modules, MoTCoder significantly improves both the modularity and correctness of the generated solutions, leading to substantial relative pass@1 improvements of 12.9% on APPS and 9.43% on CodeContests. Our codes are available at https://github.com/dvlab-research/MoTCoder.

{{</citation>}}


### (48/67) BAL: Balancing Diversity and Novelty for Active Learning (Jingyao Li et al., 2023)

{{<citation>}}

Jingyao Li, Pengguang Chen, Shaozuo Yu, Shu Liu, Jiaya Jia. (2023)  
**BAL: Balancing Diversity and Novelty for Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.15944v1)  

---


**ABSTRACT**  
The objective of Active Learning is to strategically label a subset of the dataset to maximize performance within a predetermined labeling budget. In this study, we harness features acquired through self-supervised learning. We introduce a straightforward yet potent metric, Cluster Distance Difference, to identify diverse data. Subsequently, we introduce a novel framework, Balancing Active Learning (BAL), which constructs adaptive sub-pools to balance diverse and uncertain data. Our approach outperforms all established active learning methods on widely recognized benchmarks by 1.20%. Moreover, we assess the efficacy of our proposed framework under extended settings, encompassing both larger and smaller labeling budgets. Experimental results demonstrate that, when labeling 80% of the samples, the performance of the current SOTA method declines by 0.74%, whereas our proposed BAL achieves performance comparable to the full dataset. Codes are available at https://github.com/JulietLJY/BAL.

{{</citation>}}


### (49/67) Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations (Renzhe Zhou et al., 2023)

{{<citation>}}

Renzhe Zhou, Chen-Xiao Gao, Zongzhang Zhang, Yang Yu. (2023)  
**Generalizable Task Representation Learning for Offline Meta-Reinforcement Learning with Data Limitations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.15909v1)  

---


**ABSTRACT**  
Generalization and sample efficiency have been long-standing issues concerning reinforcement learning, and thus the field of Offline Meta-Reinforcement Learning~(OMRL) has gained increasing attention due to its potential of solving a wide range of problems with static and limited offline data. Existing OMRL methods often assume sufficient training tasks and data coverage to apply contrastive learning to extract task representations. However, such assumptions are not applicable in several real-world applications and thus undermine the generalization ability of the representations. In this paper, we consider OMRL with two types of data limitations: limited training tasks and limited behavior diversity and propose a novel algorithm called GENTLE for learning generalizable task representations in the face of data limitations. GENTLE employs Task Auto-Encoder~(TAE), which is an encoder-decoder architecture to extract the characteristics of the tasks. Unlike existing methods, TAE is optimized solely by reconstruction of the state transition and reward, which captures the generative structure of the task models and produces generalizable representations when training tasks are limited. To alleviate the effect of limited behavior diversity, we consistently construct pseudo-transitions to align the data distribution used to train TAE with the data distribution encountered during testing. Empirically, GENTLE significantly outperforms existing OMRL methods on both in-distribution tasks and out-of-distribution tasks across both the given-context protocol and the one-shot protocol.

{{</citation>}}


### (50/67) AdapterDistillation: Non-Destructive Task Composition with Knowledge Distillation (Junjie Wang et al., 2023)

{{<citation>}}

Junjie Wang, Yicheng Chen, Wangshu Zhang, Sen Hu, Teng Xu, Jing Zheng. (2023)  
**AdapterDistillation: Non-Destructive Task Composition with Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.16261v1)  

---


**ABSTRACT**  
Leveraging knowledge from multiple tasks through introducing a small number of task specific parameters into each transformer layer, also known as adapters, receives much attention recently. However, adding an extra fusion layer to implement knowledge composition not only increases the inference time but also is non-scalable for some applications. To avoid these issues, we propose a two-stage knowledge distillation algorithm called AdapterDistillation. In the first stage, we extract task specific knowledge by using local data to train a student adapter. In the second stage, we distill the knowledge from the existing teacher adapters into the student adapter to help its inference. Extensive experiments on frequently asked question retrieval in task-oriented dialog systems validate the efficiency of AdapterDistillation. We show that AdapterDistillation outperforms existing algorithms in terms of accuracy, resource consumption and inference time.

{{</citation>}}


### (51/67) ANN vs SNN: A case study for Neural Decoding in Implantable Brain-Machine Interfaces (Biyan Zhou et al., 2023)

{{<citation>}}

Biyan Zhou, Pao-Sheng Vincent Sun, Arindam Basu. (2023)  
**ANN vs SNN: A case study for Neural Decoding in Implantable Brain-Machine Interfaces**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs-NE, cs.LG, q-bio-NC  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.15889v1)  

---


**ABSTRACT**  
While it is important to make implantable brain-machine interfaces (iBMI) wireless to increase patient comfort and safety, the trend of increased channel count in recent neural probes poses a challenge due to the concomitant increase in the data rate. Extracting information from raw data at the source by using edge computing is a promising solution to this problem, with integrated intention decoders providing the best compression ratio. In this work, we compare different neural networks (NN) for motor decoding in terms of accuracy and implementation cost. We further show that combining traditional signal processing techniques with machine learning ones deliver surprisingly good performance even with simple NNs. Adding a block Bidirectional Bessel filter provided maximum gains of $\approx 0.05$, $0.04$ and $0.03$ in $R^2$ for ANN\_3d, SNN\_3D and ANN models, while the gains were lower ($\approx 0.02$ or less) for LSTM and SNN\_streaming models. Increasing training data helped improve the $R^2$ of all models by $0.03-0.04$ indicating they have more capacity for future improvement. In general, LSTM and SNN\_streaming models occupy the high and low ends of the pareto curves (for accuracy vs. memory/operations) respectively while SNN\_3D and ANN\_3D occupy intermediate positions. Our work presents state of the art results for this dataset and paves the way for decoder-integrated-implants of the future.

{{</citation>}}


### (52/67) PDiT: Interleaving Perception and Decision-making Transformers for Deep Reinforcement Learning (Hangyu Mao et al., 2023)

{{<citation>}}

Hangyu Mao, Rui Zhao, Ziyue Li, Zhiwei Xu, Hao Chen, Yiqun Chen, Bin Zhang, Zhen Xiao, Junge Zhang, Jiangjin Yin. (2023)  
**PDiT: Interleaving Perception and Decision-making Transformers for Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.15863v1)  

---


**ABSTRACT**  
Designing better deep networks and better reinforcement learning (RL) algorithms are both important for deep RL. This work studies the former. Specifically, the Perception and Decision-making Interleaving Transformer (PDiT) network is proposed, which cascades two Transformers in a very natural way: the perceiving one focuses on \emph{the environmental perception} by processing the observation at the patch level, whereas the deciding one pays attention to \emph{the decision-making} by conditioning on the history of the desired returns, the perceiver's outputs, and the actions. Such a network design is generally applicable to a lot of deep RL settings, e.g., both the online and offline RL algorithms under environments with either image observations, proprioception observations, or hybrid image-language observations. Extensive experiments show that PDiT can not only achieve superior performance than strong baselines in different settings but also extract explainable feature representations. Our code is available at \url{https://github.com/maohangyu/PDiT}.

{{</citation>}}


### (53/67) Curricular and Cyclical Loss for Time Series Learning Strategy (Chenxi Sun et al., 2023)

{{<citation>}}

Chenxi Sun, Hongyan Li, Moxian Song, Derun Cai, Shenda Hong. (2023)  
**Curricular and Cyclical Loss for Time Series Learning Strategy**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.15853v1)  

---


**ABSTRACT**  
Time series widely exists in real-world applications and many deep learning models have performed well on it. Current research has shown the importance of learning strategy for models, suggesting that the benefit is the order and size of learning samples. However, no effective strategy has been proposed for time series due to its abstract and dynamic construction. Meanwhile, the existing one-shot tasks and continuous tasks for time series necessitate distinct learning processes and mechanisms. No all-purpose approach has been suggested. In this work, we propose a novel Curricular and CyclicaL loss (CRUCIAL) to learn time series for the first time. It is model- and task-agnostic and can be plugged on top of the original loss with no extra procedure. CRUCIAL has two characteristics: It can arrange an easy-to-hard learning order by dynamically determining the sample contribution and modulating the loss amplitude; It can manage a cyclically changed dataset and achieve an adaptive cycle by correlating the loss distribution and the selection probability. We prove that compared with monotonous size, cyclical size can reduce expected error. Experiments on 3 kinds of tasks and 5 real-world datasets show the benefits of CRUCIAL for most deep learning models when learning time series.

{{</citation>}}


## cs.IR (6)



### (54/67) Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages (Mofetoluwa Adeyemi et al., 2023)

{{<citation>}}

Mofetoluwa Adeyemi, Akintunde Oladipo, Ronak Pradeep, Jimmy Lin. (2023)  
**Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, Low-Resource, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.16159v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown impressive zero-shot capabilities in various document reranking tasks. Despite their successful implementations, there is still a gap in existing literature on their effectiveness in low-resource languages. To address this gap, we investigate how LLMs function as rerankers in cross-lingual information retrieval (CLIR) systems for African languages. Our implementation covers English and four African languages (Hausa, Somali, Swahili, and Yoruba) and we examine cross-lingual reranking with queries in English and passages in the African languages. Additionally, we analyze and compare the effectiveness of monolingual reranking using both query and document translations. We also evaluate the effectiveness of LLMs when leveraging their own generated translations. To get a grasp of the effectiveness of multiple LLMs, our study focuses on the proprietary models RankGPT-4 and RankGPT-3.5, along with the open-source model, RankZephyr. While reranking remains most effective in English, our results reveal that cross-lingual reranking may be competitive with reranking in African languages depending on the multilingual capability of the LLM.

{{</citation>}}


### (55/67) Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models (Manveer Singh Tamber et al., 2023)

{{<citation>}}

Manveer Singh Tamber, Ronak Pradeep, Jimmy Lin. (2023)  
**Scaling Down, LiTting Up: Efficient Zero-Shot Listwise Reranking with Seq2seq Encoder-Decoder Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: T5, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.16098v1)  

---


**ABSTRACT**  
Recent work in zero-shot listwise reranking using LLMs has achieved state-of-the-art results. However, these methods are not without drawbacks. The proposed methods rely on large LLMs with billions of parameters and limited context sizes. This paper introduces LiT5-Distill and LiT5-Score, two methods for efficient zero-shot listwise reranking, leveraging T5 sequence-to-sequence encoder-decoder models. Our approaches demonstrate competitive reranking effectiveness compared to recent state-of-the-art LLM rerankers with substantially smaller models. Through LiT5-Score, we also explore the use of cross-attention to calculate relevance scores to perform reranking, eliminating the reliance on external passage relevance labels for training. We present a range of models from 220M parameters to 3B parameters, all with strong reranking results, challenging the necessity of large-scale models for effective zero-shot reranking and opening avenues for more efficient listwise reranking solutions. We provide code and scripts to reproduce our results at https://github.com/castorini/LiT5.

{{</citation>}}


### (56/67) Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via Large Language Models (Fan Liu et al., 2023)

{{<citation>}}

Fan Liu, Yaqi Liu, Zhiyong Cheng, Liqiang Nie, Mohan Kankanhalli. (2023)  
**Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via Large Language Models**  

---
Primary Category: cs.IR  
Categories: H-3-3, cs-IR, cs-MM, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16275v1)  

---


**ABSTRACT**  
Recommendation systems harness user-item interactions like clicks and reviews to learn their representations. Previous studies improve recommendation accuracy and interpretability by modeling user preferences across various aspects and intents. However, the aspects and intents are inferred directly from user reviews or behavior patterns, suffering from the data noise and the data sparsity problem. Furthermore, it is difficult to understand the reasons behind recommendations due to the challenges of interpreting implicit aspects and intents. Inspired by the deep semantic understanding offered by large language models (LLMs), we introduce a chain-based prompting approach to uncover semantic aspect-aware interactions, which provide clearer insights into user behaviors at a fine-grained semantic level. To incorporate the abundant interactions of various aspects, we propose the simple yet effective Semantic Aspect-based Graph Convolution Network (short for SAGCN). By performing graph convolutions on multiple semantic aspect graphs, SAGCN efficiently combines embeddings across multiple semantic aspects for final user and item representations. The effectiveness of the SAGCN was evaluated on three publicly available datasets through extensive experiments, which revealed that it outperforms all other competitors. Furthermore, interpretability analysis experiments were conducted to demonstrate the interpretability of incorporating semantic aspects into the model.

{{</citation>}}


### (57/67) RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation (Sichun Luo et al., 2023)

{{<citation>}}

Sichun Luo, Bowei He, Haohan Zhao, Yinya Huang, Aojun Zhou, Zongpeng Li, Yuanzhang Xiao, Mingjie Zhan, Linqi Song. (2023)  
**RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16018v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable capabilities and have been extensively deployed across various domains, including recommender systems. Numerous studies have employed specialized \textit{prompts} to harness the in-context learning capabilities intrinsic to LLMs. For example, LLMs are prompted to act as zero-shot rankers for listwise ranking, evaluating candidate items generated by a retrieval model for recommendation. Recent research further uses instruction tuning techniques to align LLM with human preference for more promising recommendations. Despite its potential, current research overlooks the integration of multiple ranking tasks to enhance model performance. Moreover, the signal from the conventional recommendation model is not integrated into the LLM, limiting the current system performance.   In this paper, we introduce RecRanker, tailored for instruction tuning LLM to serve as the \textbf{Ranker} for top-\textit{k} \textbf{Rec}ommendations. Specifically, we introduce importance-aware sampling, clustering-based sampling, and penalty for repetitive sampling for sampling high-quality, representative, and diverse training data. To enhance the prompt, we introduce position shifting strategy to mitigate position bias and augment the prompt with auxiliary information from conventional recommendation models, thereby enriching the contextual understanding of the LLM. Subsequently, we utilize the sampled data to assemble an instruction-tuning dataset with the augmented prompt comprising three distinct ranking tasks: pointwise, pairwise, and listwise rankings. We further propose a hybrid ranking method to enhance the model performance by ensembling these ranking tasks. Our empirical evaluations demonstrate the effectiveness of our proposed RecRanker in both direct and sequential recommendation scenarios.

{{</citation>}}


### (58/67) Dynamic In-Context Learning from Nearest Neighbors for Bundle Generation (Zhu Sun et al., 2023)

{{<citation>}}

Zhu Sun, Kaidong Feng, Jie Yang, Xinghua Qu, Hui Fang, Yew-Soon Ong, Wenyuan Liu. (2023)  
**Dynamic In-Context Learning from Nearest Neighbors for Bundle Generation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.16262v1)  

---


**ABSTRACT**  
Product bundling has evolved into a crucial marketing strategy in e-commerce. However, current studies are limited to generating (1) fixed-size or single bundles, and most importantly, (2) bundles that do not reflect consistent user intents, thus being less intelligible or useful to users. This paper explores two interrelated tasks, i.e., personalized bundle generation and the underlying intent inference based on users' interactions in a session, leveraging the logical reasoning capability of large language models. We introduce a dynamic in-context learning paradigm, which enables ChatGPT to seek tailored and dynamic lessons from closely related sessions as demonstrations while performing tasks in the target session. Specifically, it first harnesses retrieval augmented generation to identify nearest neighbor sessions for each target session. Then, proper prompts are designed to guide ChatGPT to perform the two tasks on neighbor sessions. To enhance reliability and mitigate the hallucination issue, we develop (1) a self-correction strategy to foster mutual improvement in both tasks without supervision signals; and (2) an auto-feedback mechanism to recurrently offer dynamic supervision based on the distinct mistakes made by ChatGPT on various neighbor sessions. Thus, the target session can receive customized and dynamic lessons for improved performance by observing the demonstrations of its neighbor sessions. Finally, experimental results on three real-world datasets verify the effectiveness of our methods on both tasks. Additionally, the inferred intents can prove beneficial for other intriguing downstream tasks, such as crafting appealing bundle names.

{{</citation>}}


### (59/67) Hypergraph Enhanced Knowledge Tree Prompt Learning for Next-Basket Recommendation (Zi-Feng Mai et al., 2023)

{{<citation>}}

Zi-Feng Mai, Chang-Dong Wang, Zhongjie Zeng, Ya Li, Jiaquan Chen, Philip S. Yu. (2023)  
**Hypergraph Enhanced Knowledge Tree Prompt Learning for Next-Basket Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.15851v1)  

---


**ABSTRACT**  
Next-basket recommendation (NBR) aims to infer the items in the next basket given the corresponding basket sequence. Existing NBR methods are mainly based on either message passing in a plain graph or transition modelling in a basket sequence. However, these methods only consider point-to-point binary item relations while item dependencies in real world scenarios are often in higher order. Additionally, the importance of the same item to different users varies due to variation of user preferences, and the relations between items usually involve various aspects. As pretrained language models (PLMs) excel in multiple tasks in natural language processing (NLP) and computer vision (CV), many researchers have made great efforts in utilizing PLMs to boost recommendation. However, existing PLM-based recommendation methods degrade when encountering Out-Of-Vocabulary (OOV) items. OOV items are those whose IDs are out of PLM's vocabulary and thus unintelligible to PLM. To settle the above challenges, we propose a novel method HEKP4NBR, which transforms the knowledge graph (KG) into prompts, namely Knowledge Tree Prompt (KTP), to help PLM encode the OOV item IDs in the user's basket sequence. A hypergraph convolutional module is designed to build a hypergraph based on item similarities measured by an MoE model from multiple aspects and then employ convolution on the hypergraph to model correlations among multiple items. Extensive experiments are conducted on HEKP4NBR on two datasets based on real company data and validate its effectiveness against multiple state-of-the-art methods.

{{</citation>}}


## cs.NI (1)



### (60/67) A Bayesian Framework of Deep Reinforcement Learning for Joint O-RAN/MEC Orchestration (Fahri Wisnu Murti et al., 2023)

{{<citation>}}

Fahri Wisnu Murti, Samad Ali, Matti Latva-aho. (2023)  
**A Bayesian Framework of Deep Reinforcement Learning for Joint O-RAN/MEC Orchestration**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16142v1)  

---


**ABSTRACT**  
Multi-access Edge Computing (MEC) can be implemented together with Open Radio Access Network (O-RAN) over commodity platforms to offer low-cost deployment and bring the services closer to end-users. In this paper, a joint O-RAN/MEC orchestration using a Bayesian deep reinforcement learning (RL)-based framework is proposed that jointly controls the O-RAN functional splits, the allocated resources and hosting locations of the O-RAN/MEC services across geo-distributed platforms, and the routing for each O-RAN/MEC data flow. The goal is to minimize the long-term overall network operation cost and maximize the MEC performance criterion while adapting possibly time-varying O-RAN/MEC demands and resource availability. This orchestration problem is formulated as Markov decision process (MDP). However, the system consists of multiple BSs that share the same resources and serve heterogeneous demands, where their parameters have non-trivial relations. Consequently, finding the exact model of the underlying system is impractical, and the formulated MDP renders in a large state space with multi-dimensional discrete action. To address such modeling and dimensionality issues, a novel model-free RL agent is proposed for our solution framework. The agent is built from Double Deep Q-network (DDQN) that tackles the large state space and is then incorporated with action branching, an action decomposition method that effectively addresses the multi-dimensional discrete action with linear increase complexity. Further, an efficient exploration-exploitation strategy under a Bayesian framework using Thomson sampling is proposed to improve the learning performance and expedite its convergence. Trace-driven simulations are performed using an O-RAN-compliant model. The results show that our approach is data-efficient (i.e., converges faster) and increases the returned reward by 32\% than its non-Bayesian version.

{{</citation>}}


## cs.AI (3)



### (61/67) Large Language Model Situational Awareness Based Planning (Liman Wang et al., 2023)

{{<citation>}}

Liman Wang, Hanyang Zhong. (2023)  
**Large Language Model Situational Awareness Based Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16127v1)  

---


**ABSTRACT**  
This work pioneers evaluating emergent planning capabilities based on situational awareness in large language models. We contribute (i) novel benchmarks and metrics for standardized assessment; (ii) a unique dataset to spur progress; and (iii) demonstrations that prompting and multi-agent schemes significantly enhance planning performance in context-sensitive planning tasks. Positioning this within a situated agent and automated planning research, we highlight inherent reliability challenges--efficiently mapping world states to actions without environmental guidance remains open despite simulated domain advances. Although out-of-scope, limitations around validation methodology and data availability indicate exciting directions, including fine-tuning on expanded planning corpora and optimizations for triggering fast latent planning. By conclusively demonstrating current methods' promise and limitations via rigorous comparison, we catalyze investigating reliable goal-directed reasoning for situated agents.

{{</citation>}}


### (62/67) Large Language Models as Traffic Signal Control Agents: Capacity and Opportunity (Siqi Lai et al., 2023)

{{<citation>}}

Siqi Lai, Zhao Xu, Weijia Zhang, Hao Liu, Hui Xiong. (2023)  
**Large Language Models as Traffic Signal Control Agents: Capacity and Opportunity**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16044v1)  

---


**ABSTRACT**  
Traffic signal control is crucial for optimizing the efficiency of road network by regulating traffic light phases. Existing research predominantly focuses on heuristic or reinforcement learning (RL)-based methods, which often lack transferability across diverse traffic scenarios and suffer from poor interpretability. This paper introduces a novel approach, LLMLight, utilizing large language models (LLMs) for traffic signal control tasks. By leveraging LLMs' impressive generalization and zero-shot reasoning capabilities, LLMLight executes a human-like decision-making process for efficient traffic management. Specifically, the framework begins by composing task descriptions, current traffic conditions, and prior knowledge into a prompt. Subsequently, we utilize LLM's chain-of-thought (CoT) reasoning ability to identify the next traffic signal phase, ensuring optimal efficiency in the road network. LLMLight achieves state-of-the-art (SOTA) or competitive results across five real-world traffic datasets. Notably, LLMLight showcases remarkable generalization, interpretability, and zero-shot reasoning abilities, even without any training for transportation management tasks. Our project is available at https://github.com/usail-hkust/LLMTSCS.

{{</citation>}}


### (63/67) Decentralized Monte Carlo Tree Search for Partially Observable Multi-agent Pathfinding (Alexey Skrynnik et al., 2023)

{{<citation>}}

Alexey Skrynnik, Anton Andreychuk, Konstantin Yakovlev, Aleksandr Panov. (2023)  
**Decentralized Monte Carlo Tree Search for Partially Observable Multi-agent Pathfinding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15908v1)  

---


**ABSTRACT**  
The Multi-Agent Pathfinding (MAPF) problem involves finding a set of conflict-free paths for a group of agents confined to a graph. In typical MAPF scenarios, the graph and the agents' starting and ending vertices are known beforehand, allowing the use of centralized planning algorithms. However, in this study, we focus on the decentralized MAPF setting, where the agents may observe the other agents only locally and are restricted in communications with each other. Specifically, we investigate the lifelong variant of MAPF, where new goals are continually assigned to the agents upon completion of previous ones. Drawing inspiration from the successful AlphaZero approach, we propose a decentralized multi-agent Monte Carlo Tree Search (MCTS) method for MAPF tasks. Our approach utilizes the agent's observations to recreate the intrinsic Markov decision process, which is then used for planning with a tailored for multi-agent tasks version of neural MCTS. The experimental results show that our approach outperforms state-of-the-art learnable MAPF solvers. The source code is available at https://github.com/AIRI-Institute/mats-lp.

{{</citation>}}


## cs.RO (2)



### (64/67) Coordination and Machine Learning in Multi-Robot Systems: Applications in Robotic Soccer (Luis Paulo Reis, 2023)

{{<citation>}}

Luis Paulo Reis. (2023)  
**Coordination and Machine Learning in Multi-Robot Systems: Applications in Robotic Soccer**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16273v1)  

---


**ABSTRACT**  
This paper presents the concepts of Artificial Intelligence, Multi-Agent-Systems, Coordination, Intelligent Robotics and Deep Reinforcement Learning. Emphasis is given on and how AI and DRL, may be efficiently used to create efficient robot skills and coordinated robotic teams, capable of performing very complex actions and tasks, such as playing a game of soccer. The paper also presents the concept of robotic soccer and the vision and structure of the RoboCup initiative with emphasis on the Humanoid Simulation 3D league and the new challenges this competition, poses. The final topics presented at the paper are based on the research developed/coordinated by the author throughout the last 22 years in the context of the FCPortugal project. The paper presents a short description of the coordination methodologies developed, such as: Strategy, Tactics, Formations, Setplays, and Coaching Languages and the use of Machine Learning to optimize the use of this concepts. The topics presented also include novel stochastic search algorithms for black box optimization and their use in the optimization of omnidirectional walking skills, robotic multi-agent learning and the creation of a humanoid kick with controlled distance. Finally, new applications using variations of the Proximal Policy Optimization algorithm and advanced modelling for robot and multi-robot learning are briefly explained with emphasis for our new humanoid sprinting and running skills and an amazing humanoid robot soccer dribbling skill. FCPortugal project enabled us to publish more than 100 papers and win several competitions in different leagues and many scientific awards at RoboCup. In total, our team won more than 40 awards in international competitions including a clear victory at the Simulation 3D League at RoboCup 2022 competition, scoring 84 goals and conceding only 2.

{{</citation>}}


### (65/67) V-STRONG: Visual Self-Supervised Traversability Learning for Off-road Navigation (Sanghun Jung et al., 2023)

{{<citation>}}

Sanghun Jung, JoonHo Lee, Xiangyun Meng, Byron Boots, Alexander Lambert. (2023)  
**V-STRONG: Visual Self-Supervised Traversability Learning for Off-road Navigation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.16016v1)  

---


**ABSTRACT**  
Reliable estimation of terrain traversability is critical for the successful deployment of autonomous systems in wild, outdoor environments. Given the lack of large-scale annotated datasets for off-road navigation, strictly-supervised learning approaches remain limited in their generalization ability. To this end, we introduce a novel, image-based self-supervised learning method for traversability prediction, leveraging a state-of-the-art vision foundation model for improved out-of-distribution performance. Our method employs contrastive representation learning using both human driving data and instance-based segmentation masks during training. We show that this simple, yet effective, technique drastically outperforms recent methods in predicting traversability for both on- and off-trail driving scenarios. We compare our method with recent baselines on both a common benchmark as well as our own datasets, covering a diverse range of outdoor environments and varied terrain types. We also demonstrate the compatibility of resulting costmap predictions with a model-predictive controller. Finally, we evaluate our approach on zero- and few-shot tasks, demonstrating unprecedented performance for generalization to new environments. Videos and additional material can be found here: \url{https://sites.google.com/view/visual-traversability-learning}.

{{</citation>}}


## cs.CY (1)



### (66/67) Can ChatGPT Read Who You Are? (Erik Derner et al., 2023)

{{<citation>}}

Erik Derner, Dalibor Kučera, Nuria Oliver, Jan Zahálka. (2023)  
**Can ChatGPT Read Who You Are?**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs-HC, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.16070v1)  

---


**ABSTRACT**  
The interplay between artificial intelligence (AI) and psychology, particularly in personality assessment, represents an important emerging area of research. Accurate personality trait estimation is crucial not only for enhancing personalization in human-computer interaction but also for a wide variety of applications ranging from mental health to education. This paper analyzes the capability of a generic chatbot, ChatGPT, to effectively infer personality traits from short texts. We report the results of a comprehensive user study featuring texts written in Czech by a representative population sample of 155 participants. Their self-assessments based on the Big Five Inventory (BFI) questionnaire serve as the ground truth. We compare the personality trait estimations made by ChatGPT against those by human raters and report ChatGPT's competitive performance in inferring personality traits from text. We also uncover a 'positivity bias' in ChatGPT's assessments across all personality dimensions and explore the impact of prompt composition on accuracy. This work contributes to the understanding of AI capabilities in psychological assessment, highlighting both the potential and limitations of using large language models for personality inference. Our research underscores the importance of responsible AI development, considering ethical implications such as privacy, consent, autonomy, and bias in AI applications.

{{</citation>}}


## cs.SE (1)



### (67/67) A Prompt Learning Framework for Source Code Summarization (Weisong Sun et al., 2023)

{{<citation>}}

Weisong Sun, Chunrong Fang, Yudu You, Yuchen Chen, Yi Liu, Chong Wang, Jian Zhang, Quanjun Zhang, Hanwei Qian, Wei Zhao, Yang Liu, Zhenyu Chen. (2023)  
**A Prompt Learning Framework for Source Code Summarization**  

---
Primary Category: cs.SE  
Categories: 68-04, 68T30, D-2-3; I-2-2; I-2-4, cs-AI, cs-SE, cs.SE  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.16066v1)  

---


**ABSTRACT**  
(Source) code summarization is the task of automatically generating natural language summaries for given code snippets. Such summaries play a key role in helping developers understand and maintain source code. Recently, with the successful application of large language models (LLMs) in numerous fields, software engineering researchers have also attempted to adapt LLMs to solve code summarization tasks. The main adaptation schemes include instruction prompting and task-oriented fine-tuning. However, instruction prompting involves designing crafted prompts for zero-shot learning or selecting appropriate samples for few-shot learning and requires users to have professional domain knowledge, while task-oriented fine-tuning requires high training costs. In this paper, we propose a novel prompt learning framework for code summarization called PromptCS. PromptCS trains a prompt agent that can generate continuous prompts to unleash the potential for LLMs in code summarization. Compared to the human-written discrete prompt, the continuous prompts are produced under the guidance of LLMs and are therefore easier to understand by LLMs. PromptCS freezes the parameters of LLMs when training the prompt agent, which can greatly reduce the requirements for training resources. We evaluate PromptCS on the CodeSearchNet dataset involving multiple programming languages. The results show that PromptCS significantly outperforms instruction prompting schemes on all four widely used metrics. In some base LLMs, e.g., CodeGen-Multi-2B and StarCoderBase-1B and -3B, PromptCS even outperforms the task-oriented fine-tuning scheme. More importantly, the training efficiency of PromptCS is faster than the task-oriented fine-tuning scheme, with a more pronounced advantage on larger LLMs. The results of the human evaluation demonstrate that PromptCS can generate more good summaries compared to baselines.

{{</citation>}}
