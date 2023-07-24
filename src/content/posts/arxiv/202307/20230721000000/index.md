---
draft: false
title: "arXiv @ 2023.07.21"
date: 2023-07-21
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.21"
    identifier: arxiv_20230721
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (33)](#cscv-33)
- [cs.LG (13)](#cslg-13)
- [cs.CR (3)](#cscr-3)
- [cs.IR (6)](#csir-6)
- [cs.CL (17)](#cscl-17)
- [cs.HC (1)](#cshc-1)
- [cs.AI (7)](#csai-7)
- [stat.ML (1)](#statml-1)
- [cs.SE (4)](#csse-4)
- [cs.RO (3)](#csro-3)
- [cs.SI (3)](#cssi-3)
- [eess.AS (1)](#eessas-1)
- [eess.SY (1)](#eesssy-1)
- [cs.NI (3)](#csni-3)
- [cs.CY (2)](#cscy-2)
- [math.NA (1)](#mathna-1)
- [eess.SP (2)](#eesssp-2)
- [q-fin.TR (1)](#q-fintr-1)
- [cs.SD (3)](#cssd-3)
- [cs.DL (1)](#csdl-1)

## cs.CV (33)



### (1/106) Mining Conditional Part Semantics with Occluded Extrapolation for Human-Object Interaction Detection (Guangzhi Wang et al., 2023)

{{<citation>}}

Guangzhi Wang, Yangyang Guo, Mohan Kankanhalli. (2023)  
**Mining Conditional Part Semantics with Occluded Extrapolation for Human-Object Interaction Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.10499v1)  

---


**ABSTRACT**  
Human-Object Interaction Detection is a crucial aspect of human-centric scene understanding, with important applications in various domains. Despite recent progress in this field, recognizing subtle and detailed interactions remains challenging. Existing methods try to use human-related clues to alleviate the difficulty, but rely heavily on external annotations or knowledge, limiting their practical applicability in real-world scenarios. In this work, we propose a novel Part Semantic Network (PSN) to solve this problem. The core of PSN is a Conditional Part Attention (CPA) mechanism, where human features are taken as keys and values, and the object feature is used as query for the computation in a cross-attention mechanism. In this way, our model learns to automatically focus on the most informative human parts conditioned on the involved object, generating more semantically meaningful features for interaction recognition. Additionally, we propose an Occluded Part Extrapolation (OPE) strategy to facilitate interaction recognition under occluded scenarios, which teaches the model to extrapolate detailed features from partially occluded ones. Our method consistently outperforms prior approaches on the V-COCO and HICO-DET datasets, without external data or extra annotations. Additional ablation studies validate the effectiveness of each component of our proposed method.

{{</citation>}}


### (2/106) Backdoor Attack against Object Detection with Clean Annotation (Yize Cheng et al., 2023)

{{<citation>}}

Yize Cheng, Wenbin Hu, Minhao Cheng. (2023)  
**Backdoor Attack against Object Detection with Clean Annotation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.10487v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have shown unprecedented success in object detection tasks. However, it was also discovered that DNNs are vulnerable to multiple kinds of attacks, including Backdoor Attacks. Through the attack, the attacker manages to embed a hidden backdoor into the DNN such that the model behaves normally on benign data samples, but makes attacker-specified judgments given the occurrence of a predefined trigger. Although numerous backdoor attacks have been experimented on image classification, backdoor attacks on object detection tasks have not been properly investigated and explored. As object detection has been adopted as an important module in multiple security-sensitive applications such as autonomous driving, backdoor attacks on object detection could pose even more severe threats. Inspired by the inherent property of deep learning-based object detectors, we propose a simple yet effective backdoor attack method against object detection without modifying the ground truth annotations, specifically focusing on the object disappearance attack and object generation attack. Extensive experiments and ablation studies prove the effectiveness of our attack on two benchmark object detection datasets, PASCAL VOC07+12 and MSCOCO, on which we achieve an attack success rate of more than 92% with a poison rate of only 5%.

{{</citation>}}


### (3/106) Explaining Autonomous Driving Actions with Visual Question Answering (Shahin Atakishiyev et al., 2023)

{{<citation>}}

Shahin Atakishiyev, Mohammad Salameh, Housam Babiker, Randy Goebel. (2023)  
**Explaining Autonomous Driving Actions with Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.10408v1)  

---


**ABSTRACT**  
The end-to-end learning ability of self-driving vehicles has achieved significant milestones over the last decade owing to rapid advances in deep learning and computer vision algorithms. However, as autonomous driving technology is a safety-critical application of artificial intelligence (AI), road accidents and established regulatory principles necessitate the need for the explainability of intelligent action choices for self-driving vehicles. To facilitate interpretability of decision-making in autonomous driving, we present a Visual Question Answering (VQA) framework, which explains driving actions with question-answering-based causal reasoning. To do so, we first collect driving videos in a simulation environment using reinforcement learning (RL) and extract consecutive frames from this log data uniformly for five selected action categories. Further, we manually annotate the extracted frames using question-answer pairs as justifications for the actions chosen in each scenario. Finally, we evaluate the correctness of the VQA-predicted answers for actions on unseen driving scenes. The empirical results suggest that the VQA mechanism can provide support to interpret real-time decisions of autonomous vehicles and help enhance overall driving safety.

{{</citation>}}


### (4/106) Interpreting and Correcting Medical Image Classification with PIP-Net (Meike Nauta et al., 2023)

{{<citation>}}

Meike Nauta, Johannes H. Hegeman, Jeroen Geerdink, Jörg Schlötterer, Maurice van Keulen, Christin Seifert. (2023)  
**Interpreting and Correcting Medical Image Classification with PIP-Net**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2307.10404v1)  

---


**ABSTRACT**  
Part-prototype models are explainable-by-design image classifiers, and a promising alternative to black box AI. This paper explores the applicability and potential of interpretable machine learning, in particular PIP-Net, for automated diagnosis support on real-world medical imaging data. PIP-Net learns human-understandable prototypical image parts and we evaluate its accuracy and interpretability for fracture detection and skin cancer diagnosis. We find that PIP-Net's decision making process is in line with medical classification standards, while only provided with image-level class labels. Because of PIP-Net's unsupervised pretraining of prototypes, data quality problems such as undesired text in an X-ray or labelling errors can be easily identified. Additionally, we are the first to show that humans can manually correct the reasoning of PIP-Net by directly disabling undesired prototypes. We conclude that part-prototype models are promising for medical applications due to their interpretability and potential for advanced model debugging.

{{</citation>}}


### (5/106) TokenFlow: Consistent Diffusion Features for Consistent Video Editing (Michal Geyer et al., 2023)

{{<citation>}}

Michal Geyer, Omer Bar-Tal, Shai Bagon, Tali Dekel. (2023)  
**TokenFlow: Consistent Diffusion Features for Consistent Video Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10373v1)  

---


**ABSTRACT**  
The generative AI revolution has recently expanded to videos. Nevertheless, current state-of-the-art video models are still lagging behind image models in terms of visual quality and user control over the generated content. In this work, we present a framework that harnesses the power of a text-to-image diffusion model for the task of text-driven video editing. Specifically, given a source video and a target text-prompt, our method generates a high-quality video that adheres to the target text, while preserving the spatial layout and motion of the input video. Our method is based on a key observation that consistency in the edited video can be obtained by enforcing consistency in the diffusion feature space. We achieve this by explicitly propagating diffusion features based on inter-frame correspondences, readily available in the model. Thus, our framework does not require any training or fine-tuning, and can work in conjunction with any off-the-shelf text-to-image editing method. We demonstrate state-of-the-art editing results on a variety of real-world videos. Webpage: https://diffusion-tokenflow.github.io/

{{</citation>}}


### (6/106) Adversarial Latent Autoencoder with Self-Attention for Structural Image Synthesis (Jiajie Fan et al., 2023)

{{<citation>}}

Jiajie Fan, Laure Vuaille, Hao Wang, Thomas Bäck. (2023)  
**Adversarial Latent Autoencoder with Self-Attention for Structural Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CE, cs-CV, cs.CV, eess-IV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2307.10166v1)  

---


**ABSTRACT**  
Generative Engineering Design approaches driven by Deep Generative Models (DGM) have been proposed to facilitate industrial engineering processes. In such processes, designs often come in the form of images, such as blueprints, engineering drawings, and CAD models depending on the level of detail. DGMs have been successfully employed for synthesis of natural images, e.g., displaying animals, human faces and landscapes. However, industrial design images are fundamentally different from natural scenes in that they contain rich structural patterns and long-range dependencies, which are challenging for convolution-based DGMs to generate. Moreover, DGM-driven generation process is typically triggered based on random noisy inputs, which outputs unpredictable samples and thus cannot perform an efficient industrial design exploration. We tackle these challenges by proposing a novel model Self-Attention Adversarial Latent Autoencoder (SA-ALAE), which allows generating feasible design images of complex engineering parts. With SA-ALAE, users can not only explore novel variants of an existing design, but also control the generation process by operating in latent space. The potential of SA-ALAE is shown by generating engineering blueprints in a real automotive design task.

{{</citation>}}


### (7/106) Drone navigation and license place detection for vehicle location in indoor spaces (Moa Arvidsson et al., 2023)

{{<citation>}}

Moa Arvidsson, Sithichot Sawirot, Cristofer Englund, Fernando Alonso-Fernandez, Martin Torstensson, Boris Duran. (2023)  
**Drone navigation and license place detection for vehicle location in indoor spaces**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.10165v2)  

---


**ABSTRACT**  
Millions of vehicles are transported every year, tightly parked in vessels or boats. To reduce the risks of associated safety issues like fires, knowing the location of vehicles is essential, since different vehicles may need different mitigation measures, e.g. electric cars. This work is aimed at creating a solution based on a nano-drone that navigates across rows of parked vehicles and detects their license plates. We do so via a wall-following algorithm, and a CNN trained to detect license plates. All computations are done in real-time on the drone, which just sends position and detected images that allow the creation of a 2D map with the position of the plates. Our solution is capable of reading all plates across eight test cases (with several rows of plates, different drone speeds, or low light) by aggregation of measurements across several drone journeys.

{{</citation>}}


### (8/106) Boundary-Refined Prototype Generation: A General End-to-End Paradigm for Semi-Supervised Semantic Segmentation (Junhao Dong et al., 2023)

{{<citation>}}

Junhao Dong, Zhu Meng, Delong Liu, Zhicheng Zhao, Fei Su. (2023)  
**Boundary-Refined Prototype Generation: A General End-to-End Paradigm for Semi-Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.10097v1)  

---


**ABSTRACT**  
Prototype-based classification is a classical method in machine learning, and recently it has achieved remarkable success in semi-supervised semantic segmentation. However, the current approach isolates the prototype initialization process from the main training framework, which appears to be unnecessary. Furthermore, while the direct use of K-Means algorithm for prototype generation has considered rich intra-class variance, it may not be the optimal solution for the classification task. To tackle these problems, we propose a novel boundary-refined prototype generation (BRPG) method, which is incorporated into the whole training framework. Specifically, our approach samples and clusters high- and low-confidence features separately based on a confidence threshold, aiming to generate prototypes closer to the class boundaries. Moreover, an adaptive prototype optimization strategy is introduced to make prototype augmentation for categories with scattered feature distributions. Extensive experiments on the PASCAL VOC 2012 and Cityscapes datasets demonstrate the superiority and scalability of the proposed method, outperforming the current state-of-the-art approaches. The code is available at xxxxxxxxxxxxxx.

{{</citation>}}


### (9/106) Divert More Attention to Vision-Language Object Tracking (Mingzhe Guo et al., 2023)

{{<citation>}}

Mingzhe Guo, Zhipeng Zhang, Liping Jing, Haibin Ling, Heng Fan. (2023)  
**Divert More Attention to Vision-Language Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.10046v1)  

---


**ABSTRACT**  
Multimodal vision-language (VL) learning has noticeably pushed the tendency toward generic intelligence owing to emerging large foundation models. However, tracking, as a fundamental vision problem, surprisingly enjoys less bonus from recent flourishing VL learning. We argue that the reasons are two-fold: the lack of large-scale vision-language annotated videos and ineffective vision-language interaction learning of current works. These nuisances motivate us to design more effective vision-language representation for tracking, meanwhile constructing a large database with language annotation for model learning. Particularly, in this paper, we first propose a general attribute annotation strategy to decorate videos in six popular tracking benchmarks, which contributes a large-scale vision-language tracking database with more than 23,000 videos. We then introduce a novel framework to improve tracking by learning a unified-adaptive VL representation, where the cores are the proposed asymmetric architecture search and modality mixer (ModaMixer). To further improve VL representation, we introduce a contrastive loss to align different modalities. To thoroughly evidence the effectiveness of our method, we integrate the proposed framework on three tracking methods with different designs, i.e., the CNN-based SiamCAR, the Transformer-based OSTrack, and the hybrid structure TransT. The experiments demonstrate that our framework can significantly improve all baselines on six benchmarks. Besides empirical results, we theoretically analyze our approach to show its rationality. By revealing the potential of VL representation, we expect the community to divert more attention to VL tracking and hope to open more possibilities for future tracking with diversified multimodal messages.

{{</citation>}}


### (10/106) Class Attention to Regions of Lesion for Imbalanced Medical Image Recognition (Jia-Xin Zhuang et al., 2023)

{{<citation>}}

Jia-Xin Zhuang, Jiabin Cai, Jianguo Zhang, Wei-shi Zheng, Ruixuan Wang. (2023)  
**Class Attention to Regions of Lesion for Imbalanced Medical Image Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.10036v2)  

---


**ABSTRACT**  
Automated medical image classification is the key component in intelligent diagnosis systems. However, most medical image datasets contain plenty of samples of common diseases and just a handful of rare ones, leading to major class imbalances. Currently, it is an open problem in intelligent diagnosis to effectively learn from imbalanced training data. In this paper, we propose a simple yet effective framework, named \textbf{C}lass \textbf{A}ttention to \textbf{RE}gions of the lesion (CARE), to handle data imbalance issues by embedding attention into the training process of \textbf{C}onvolutional \textbf{N}eural \textbf{N}etworks (CNNs). The proposed attention module helps CNNs attend to lesion regions of rare diseases, therefore helping CNNs to learn their characteristics more effectively. In addition, this attention module works only during the training phase and does not change the architecture of the original network, so it can be directly combined with any existing CNN architecture. The CARE framework needs bounding boxes to represent the lesion regions of rare diseases. To alleviate the need for manual annotation, we further developed variants of CARE by leveraging the traditional saliency methods or a pretrained segmentation model for bounding box generation. Results show that the CARE variants with automated bounding box generation are comparable to the original CARE framework with \textit{manual} bounding box annotations. A series of experiments on an imbalanced skin image dataset and a pneumonia dataset indicates that our method can effectively help the network focus on the lesion regions of rare diseases and remarkably improves the classification performance of rare diseases.

{{</citation>}}


### (11/106) Towards Fair Face Verification: An In-depth Analysis of Demographic Biases (Ioannis Sarridis et al., 2023)

{{<citation>}}

Ioannis Sarridis, Christos Koutlis, Symeon Papadopoulos, Christos Diou. (2023)  
**Towards Fair Face Verification: An In-depth Analysis of Demographic Biases**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.10011v1)  

---


**ABSTRACT**  
Deep learning-based person identification and verification systems have remarkably improved in terms of accuracy in recent years; however, such systems, including widely popular cloud-based solutions, have been found to exhibit significant biases related to race, age, and gender, a problem that requires in-depth exploration and solutions. This paper presents an in-depth analysis, with a particular emphasis on the intersectionality of these demographic factors. Intersectional bias refers to the performance discrepancies w.r.t. the different combinations of race, age, and gender groups, an area relatively unexplored in current literature. Furthermore, the reliance of most state-of-the-art approaches on accuracy as the principal evaluation metric often masks significant demographic disparities in performance. To counter this crucial limitation, we incorporate five additional metrics in our quantitative analysis, including disparate impact and mistreatment metrics, which are typically ignored by the relevant fairness-aware approaches. Results on the Racial Faces in-the-Wild (RFW) benchmark indicate pervasive biases in face recognition systems, extending beyond race, with different demographic factors yielding significantly disparate outcomes. In particular, Africans demonstrate an 11.25% lower True Positive Rate (TPR) compared to Caucasians, while only a 3.51% accuracy drop is observed. Even more concerning, the intersections of multiple protected groups, such as African females over 60 years old, demonstrate a +39.89% disparate mistreatment rate compared to the highest Caucasians rate. By shedding light on these biases and their implications, this paper aims to stimulate further research towards developing fairer, more equitable face recognition and verification systems.

{{</citation>}}


### (12/106) MODA: Mapping-Once Audio-driven Portrait Animation with Dual Attentions (Yunfei Liu et al., 2023)

{{<citation>}}

Yunfei Liu, Lijian Lin, Fei Yu, Changyin Zhou, Yu Li. (2023)  
**MODA: Mapping-Once Audio-driven Portrait Animation with Dual Attentions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.10008v1)  

---


**ABSTRACT**  
Audio-driven portrait animation aims to synthesize portrait videos that are conditioned by given audio. Animating high-fidelity and multimodal video portraits has a variety of applications. Previous methods have attempted to capture different motion modes and generate high-fidelity portrait videos by training different models or sampling signals from given videos. However, lacking correlation learning between lip-sync and other movements (e.g., head pose/eye blinking) usually leads to unnatural results. In this paper, we propose a unified system for multi-person, diverse, and high-fidelity talking portrait generation. Our method contains three stages, i.e., 1) Mapping-Once network with Dual Attentions (MODA) generates talking representation from given audio. In MODA, we design a dual-attention module to encode accurate mouth movements and diverse modalities. 2) Facial composer network generates dense and detailed face landmarks, and 3) temporal-guided renderer syntheses stable videos. Extensive evaluations demonstrate that the proposed system produces more natural and realistic video portraits compared to previous methods.

{{</citation>}}


### (13/106) TbExplain: A Text-based Explanation Method for Scene Classification Models with the Statistical Prediction Correction (Amirhossein Aminimehr et al., 2023)

{{<citation>}}

Amirhossein Aminimehr, Pouya Khani, Amirali Molaei, Amirmohammad Kazemeini, Erik Cambria. (2023)  
**TbExplain: A Text-based Explanation Method for Scene Classification Models with the Statistical Prediction Correction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10003v1)  

---


**ABSTRACT**  
The field of Explainable Artificial Intelligence (XAI) aims to improve the interpretability of black-box machine learning models. Building a heatmap based on the importance value of input features is a popular method for explaining the underlying functions of such models in producing their predictions. Heatmaps are almost understandable to humans, yet they are not without flaws. Non-expert users, for example, may not fully understand the logic of heatmaps (the logic in which relevant pixels to the model's prediction are highlighted with different intensities or colors). Additionally, objects and regions of the input image that are relevant to the model prediction are frequently not entirely differentiated by heatmaps. In this paper, we propose a framework called TbExplain that employs XAI techniques and a pre-trained object detector to present text-based explanations of scene classification models. Moreover, TbExplain incorporates a novel method to correct predictions and textually explain them based on the statistics of objects in the input image when the initial prediction is unreliable. To assess the trustworthiness and validity of the text-based explanations, we conducted a qualitative experiment, and the findings indicated that these explanations are sufficiently reliable. Furthermore, our quantitative and qualitative experiments on TbExplain with scene classification datasets reveal an improvement in classification accuracy over ResNet variants.

{{</citation>}}


### (14/106) Mitigating Viewer Impact from Disturbing Imagery using AI Filters: A User-Study (Ioannis Sarridis et al., 2023)

{{<citation>}}

Ioannis Sarridis, Jochen Spangenberg, Olga Papadopoulou, Symeon Papadopoulos. (2023)  
**Mitigating Viewer Impact from Disturbing Imagery using AI Filters: A User-Study**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10334v1)  

---


**ABSTRACT**  
Exposure to disturbing imagery can significantly impact individuals, especially professionals who encounter such content as part of their work. This paper presents a user study, involving 107 participants, predominantly journalists and human rights investigators, that explores the capability of Artificial Intelligence (AI)-based image filters to potentially mitigate the emotional impact of viewing such disturbing content. We tested five different filter styles, both traditional (Blurring and Partial Blurring) and AI-based (Drawing, Colored Drawing, and Painting), and measured their effectiveness in terms of conveying image information while reducing emotional distress. Our findings suggest that the AI-based Drawing style filter demonstrates the best performance, offering a promising solution for reducing negative feelings (-30.38%) while preserving the interpretability of the image (97.19%). Despite the requirement for many professionals to eventually inspect the original images, participants suggested potential strategies for integrating AI filters into their workflow, such as using AI filters as an initial, preparatory step before viewing the original image. Overall, this paper contributes to the development of a more ethically considerate and effective visual environment for professionals routinely engaging with potentially disturbing imagery.

{{</citation>}}


### (15/106) TUNeS: A Temporal U-Net with Self-Attention for Video-based Surgical Phase Recognition (Isabel Funke et al., 2023)

{{<citation>}}

Isabel Funke, Dominik Rivoir, Stefanie Krell, Stefanie Speidel. (2023)  
**TUNeS: A Temporal U-Net with Self-Attention for Video-based Surgical Phase Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, LSTM, Self-Attention  
[Paper Link](http://arxiv.org/abs/2307.09997v1)  

---


**ABSTRACT**  
To enable context-aware computer assistance in the operating room of the future, cognitive systems need to understand automatically which surgical phase is being performed by the medical team. The primary source of information for surgical phase recognition is typically video, which presents two challenges: extracting meaningful features from the video stream and effectively modeling temporal information in the sequence of visual features. For temporal modeling, attention mechanisms have gained popularity due to their ability to capture long-range dependencies. In this paper, we explore design choices for attention in existing temporal models for surgical phase recognition and propose a novel approach that does not resort to local attention or regularization of attention weights: TUNeS is an efficient and simple temporal model that incorporates self-attention at the coarsest stage of a U-Net-like structure. In addition, we propose to train the feature extractor, a standard CNN, together with an LSTM on preferably long video segments, i.e., with long temporal context. In our experiments, all temporal models performed better on top of feature extractors that were trained with longer temporal context. On top of these contextualized features, TUNeS achieves state-of-the-art results on Cholec80.

{{</citation>}}


### (16/106) U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation (Steven Landgraf et al., 2023)

{{<citation>}}

Steven Landgraf, Markus Hillemann, Kira Wursthorn, Markus Ulrich. (2023)  
**U-CE: Uncertainty-aware Cross-Entropy for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.09947v1)  

---


**ABSTRACT**  
Deep neural networks have shown exceptional performance in various tasks, but their lack of robustness, reliability, and tendency to be overconfident pose challenges for their deployment in safety-critical applications like autonomous driving. In this regard, quantifying the uncertainty inherent to a model's prediction is a promising endeavour to address these shortcomings. In this work, we present a novel Uncertainty-aware Cross-Entropy loss (U-CE) that incorporates dynamic predictive uncertainties into the training process by pixel-wise weighting of the well-known cross-entropy loss (CE). Through extensive experimentation, we demonstrate the superiority of U-CE over regular CE training on two benchmark datasets, Cityscapes and ACDC, using two common backbone architectures, ResNet-18 and ResNet-101. With U-CE, we manage to train models that not only improve their segmentation performance but also provide meaningful uncertainties after training. Consequently, we contribute to the development of more robust and reliable segmentation models, ultimately advancing the state-of-the-art in safety-critical applications and beyond.

{{</citation>}}


### (17/106) AGAR: Attention Graph-RNN for Adaptative Motion Prediction of Point Clouds of Deformable Objects (Pedro Gomes et al., 2023)

{{<citation>}}

Pedro Gomes, Silvia Rossi, Laura Toni. (2023)  
**AGAR: Attention Graph-RNN for Adaptative Motion Prediction of Point Clouds of Deformable Objects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.09936v1)  

---


**ABSTRACT**  
This paper focuses on motion prediction for point cloud sequences in the challenging case of deformable 3D objects, such as human body motion. First, we investigate the challenges caused by deformable shapes and complex motions present in this type of representation, with the ultimate goal of understanding the technical limitations of state-of-the-art models. From this understanding, we propose an improved architecture for point cloud prediction of deformable 3D objects. Specifically, to handle deformable shapes, we propose a graph-based approach that learns and exploits the spatial structure of point clouds to extract more representative features. Then we propose a module able to combine the learned features in an adaptative manner according to the point cloud movements. The proposed adaptative module controls the composition of local and global motions for each point, enabling the network to model complex motions in deformable 3D objects more effectively. We tested the proposed method on the following datasets: MNIST moving digits, the Mixamo human bodies motions, JPEG and CWIPC-SXR real-world dynamic bodies. Simulation results demonstrate that our method outperforms the current baseline methods given its improved ability to model complex movements as well as preserve point cloud shape. Furthermore, we demonstrate the generalizability of the proposed framework for dynamic feature learning, by testing the framework for action recognition on the MSRAction3D dataset and achieving results on-par with state-of-the-art methods

{{</citation>}}


### (18/106) Embedded Heterogeneous Attention Transformer for Cross-lingual Image Captioning (Zijie Song et al., 2023)

{{<citation>}}

Zijie Song, Zhenzhen Hu, Richang Hong. (2023)  
**Embedded Heterogeneous Attention Transformer for Cross-lingual Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Attention, Image Captioning, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09915v1)  

---


**ABSTRACT**  
Cross-lingual image captioning is confronted with both cross-lingual and cross-modal challenges for multimedia analysis. The crucial issue in this task is to model the global and local matching between the image and different languages. Existing cross-modal embedding methods based on Transformer architecture oversight the local matching between the image region and monolingual words, not to mention in the face of a variety of differentiated languages. Due to the heterogeneous property of the cross-modal and cross-lingual task, we utilize the heterogeneous network to establish cross-domain relationships and the local correspondences between the image and different languages. In this paper, we propose an Embedded Heterogeneous Attention Transformer (EHAT) to build reasoning paths bridging cross-domain for cross-lingual image captioning and integrate into transformer. The proposed EHAT consists of a Masked Heterogeneous Cross-attention (MHCA), Heterogeneous Attention Reasoning Network (HARN) and Heterogeneous Co-attention (HCA). HARN as the core network, models and infers cross-domain relationship anchored by vision bounding box representation features to connect two languages word features and learn the heterogeneous maps. MHCA and HCA implement cross-domain integration in the encoder through the special heterogeneous attention and enable single model to generate two language captioning. We test on MSCOCO dataset to generate English and Chinese, which are most widely used and have obvious difference between their language families. Our experiments show that our method even achieve better than advanced monolingual methods.

{{</citation>}}


### (19/106) A reinforcement learning approach for VQA validation: an application to diabetic macular edema grading (Tatiana Fountoukidou et al., 2023)

{{<citation>}}

Tatiana Fountoukidou, Raphael Sznitman. (2023)  
**A reinforcement learning approach for VQA validation: an application to diabetic macular edema grading**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.09886v1)  

---


**ABSTRACT**  
Recent advances in machine learning models have greatly increased the performance of automated methods in medical image analysis. However, the internal functioning of such models is largely hidden, which hinders their integration in clinical practice. Explainability and trust are viewed as important aspects of modern methods, for the latter's widespread use in clinical communities. As such, validation of machine learning models represents an important aspect and yet, most methods are only validated in a limited way. In this work, we focus on providing a richer and more appropriate validation approach for highly powerful Visual Question Answering (VQA) algorithms. To better understand the performance of these methods, which answer arbitrary questions related to images, this work focuses on an automatic visual Turing test (VTT). That is, we propose an automatic adaptive questioning method, that aims to expose the reasoning behavior of a VQA algorithm. Specifically, we introduce a reinforcement learning (RL) agent that observes the history of previously asked questions, and uses it to select the next question to pose. We demonstrate our approach in the context of evaluating algorithms that automatically answer questions related to diabetic macular edema (DME) grading. The experiments show that such an agent has similar behavior to a clinician, whereby asking questions that are relevant to key clinical concepts.

{{</citation>}}


### (20/106) BSDM: Background Suppression Diffusion Model for Hyperspectral Anomaly Detection (Jitao Ma et al., 2023)

{{<citation>}}

Jitao Ma, Weiying Xie, Yunsong Li, Leyuan Fang. (2023)  
**BSDM: Background Suppression Diffusion Model for Hyperspectral Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.09861v1)  

---


**ABSTRACT**  
Hyperspectral anomaly detection (HAD) is widely used in Earth observation and deep space exploration. A major challenge for HAD is the complex background of the input hyperspectral images (HSIs), resulting in anomalies confused in the background. On the other hand, the lack of labeled samples for HSIs leads to poor generalization of existing HAD methods. This paper starts the first attempt to study a new and generalizable background learning problem without labeled samples. We present a novel solution BSDM (background suppression diffusion model) for HAD, which can simultaneously learn latent background distributions and generalize to different datasets for suppressing complex background. It is featured in three aspects: (1) For the complex background of HSIs, we design pseudo background noise and learn the potential background distribution in it with a diffusion model (DM). (2) For the generalizability problem, we apply a statistical offset module so that the BSDM adapts to datasets of different domains without labeling samples. (3) For achieving background suppression, we innovatively improve the inference process of DM by feeding the original HSIs into the denoising network, which removes the background as noise. Our work paves a new background suppression way for HAD that can improve HAD performance without the prerequisite of manually labeled data. Assessments and generalization experiments of four HAD methods on several real HSI datasets demonstrate the above three unique properties of the proposed method. The code is available at https://github.com/majitao-xd/BSDM-HAD.

{{</citation>}}


### (21/106) Blind Image Quality Assessment Using Multi-Stream Architecture with Spatial and Channel Attention (Hassan Khalid et al., 2023)

{{<citation>}}

Hassan Khalid, Nisar Ahmed. (2023)  
**Blind Image Quality Assessment Using Multi-Stream Architecture with Spatial and Channel Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention, QA  
[Paper Link](http://arxiv.org/abs/2307.09857v1)  

---


**ABSTRACT**  
BIQA (Blind Image Quality Assessment) is an important field of study that evaluates images automatically. Although significant progress has been made, blind image quality assessment remains a difficult task since images vary in content and distortions. Most algorithms generate quality without emphasizing the important region of interest. In order to solve this, a multi-stream spatial and channel attention-based algorithm is being proposed. This algorithm generates more accurate predictions with a high correlation to human perceptual assessment by combining hybrid features from two different backbones, followed by spatial and channel attention to provide high weights to the region of interest. Four legacy image quality assessment datasets are used to validate the effectiveness of our proposed approach. Authentic and synthetic distortion image databases are used to demonstrate the effectiveness of the proposed method, and we show that it has excellent generalization properties with a particular focus on the perceptual foreground information.

{{</citation>}}


### (22/106) Hierarchical Spatio-Temporal Representation Learning for Gait Recognition (Lei Wang et al., 2023)

{{<citation>}}

Lei Wang, Bo Liu, Fangfang Liang, Bincheng Wang. (2023)  
**Hierarchical Spatio-Temporal Representation Learning for Gait Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.09856v1)  

---


**ABSTRACT**  
Gait recognition is a biometric technique that identifies individuals by their unique walking styles, which is suitable for unconstrained environments and has a wide range of applications. While current methods focus on exploiting body part-based representations, they often neglect the hierarchical dependencies between local motion patterns. In this paper, we propose a hierarchical spatio-temporal representation learning (HSTL) framework for extracting gait features from coarse to fine. Our framework starts with a hierarchical clustering analysis to recover multi-level body structures from the whole body to local details. Next, an adaptive region-based motion extractor (ARME) is designed to learn region-independent motion features. The proposed HSTL then stacks multiple ARMEs in a top-down manner, with each ARME corresponding to a specific partition level of the hierarchy. An adaptive spatio-temporal pooling (ASTP) module is used to capture gait features at different levels of detail to perform hierarchical feature mapping. Finally, a frame-level temporal aggregation (FTA) module is employed to reduce redundant information in gait sequences through multi-scale temporal downsampling. Extensive experiments on CASIA-B, OUMVLP, GREW, and Gait3D datasets demonstrate that our method outperforms the state-of-the-art while maintaining a reasonable balance between model accuracy and complexity.

{{</citation>}}


### (23/106) Density-invariant Features for Distant Point Cloud Registration (Quan Liu et al., 2023)

{{<citation>}}

Quan Liu, Hongzi Zhu, Yunsong Zhou, Hongyang Li, Shan Chang, Minyi Guo. (2023)  
**Density-invariant Features for Distant Point Cloud Registration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.09788v1)  

---


**ABSTRACT**  
Registration of distant outdoor LiDAR point clouds is crucial to extending the 3D vision of collaborative autonomous vehicles, and yet is challenging due to small overlapping area and a huge disparity between observed point densities. In this paper, we propose Group-wise Contrastive Learning (GCL) scheme to extract density-invariant geometric features to register distant outdoor LiDAR point clouds. We mark through theoretical analysis and experiments that, contrastive positives should be independent and identically distributed (i.i.d.), in order to train densityinvariant feature extractors. We propose upon the conclusion a simple yet effective training scheme to force the feature of multiple point clouds in the same spatial location (referred to as positive groups) to be similar, which naturally avoids the sampling bias introduced by a pair of point clouds to conform with the i.i.d. principle. The resulting fully-convolutional feature extractor is more powerful and density-invariant than state-of-the-art methods, improving the registration recall of distant scenarios on KITTI and nuScenes benchmarks by 40.9% and 26.9%, respectively. The code will be open-sourced.

{{</citation>}}


### (24/106) DVPT: Dynamic Visual Prompt Tuning of Large Pre-trained Models for Medical Image Analysis (Along He et al., 2023)

{{<citation>}}

Along He, Kai Wang, Zhihong Wang, Tao Li, Huazhu Fu. (2023)  
**DVPT: Dynamic Visual Prompt Tuning of Large Pre-trained Models for Medical Image Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.09787v1)  

---


**ABSTRACT**  
Limited labeled data makes it hard to train models from scratch in medical domain, and an important paradigm is pre-training and then fine-tuning. Large pre-trained models contain rich representations, which can be adapted to downstream medical tasks. However, existing methods either tune all the parameters or the task-specific layers of the pre-trained models, ignoring the input variations of medical images, and thus they are not efficient or effective. In this work, we aim to study parameter-efficient fine-tuning (PEFT) for medical image analysis, and propose a dynamic visual prompt tuning method, named DVPT. It can extract knowledge beneficial to downstream tasks from large models with a few trainable parameters. Firstly, the frozen features are transformed by an lightweight bottleneck layer to learn the domain-specific distribution of downstream medical tasks, and then a few learnable visual prompts are used as dynamic queries and then conduct cross-attention with the transformed features, attempting to acquire sample-specific knowledge that are suitable for each sample. Finally, the features are projected to original feature dimension and aggregated with the frozen features. This DVPT module can be shared between different Transformer layers, further reducing the trainable parameters. To validate DVPT, we conduct extensive experiments with different pre-trained models on medical classification and segmentation tasks. We find such PEFT method can not only efficiently adapt the pre-trained models to the medical domain, but also brings data efficiency with partial labeled data. For example, with 0.5\% extra trainable parameters, our method not only outperforms state-of-the-art PEFT methods, even surpasses the full fine-tuning by more than 2.20\% Kappa score on medical classification task. It can saves up to 60\% labeled data and 99\% storage cost of ViT-B/16.

{{</citation>}}


### (25/106) Source-Free Domain Adaptation for Medical Image Segmentation via Prototype-Anchored Feature Alignment and Contrastive Learning (Qinji Yu et al., 2023)

{{<citation>}}

Qinji Yu, Nan Xi, Junsong Yuan, Ziyu Zhou, Kang Dang, Xiaowei Ding. (2023)  
**Source-Free Domain Adaptation for Medical Image Segmentation via Prototype-Anchored Feature Alignment and Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.09769v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation (UDA) has increasingly gained interests for its capacity to transfer the knowledge learned from a labeled source domain to an unlabeled target domain. However, typical UDA methods require concurrent access to both the source and target domain data, which largely limits its application in medical scenarios where source data is often unavailable due to privacy concern. To tackle the source data-absent problem, we present a novel two-stage source-free domain adaptation (SFDA) framework for medical image segmentation, where only a well-trained source segmentation model and unlabeled target data are available during domain adaptation. Specifically, in the prototype-anchored feature alignment stage, we first utilize the weights of the pre-trained pixel-wise classifier as source prototypes, which preserve the information of source features. Then, we introduce the bi-directional transport to align the target features with class prototypes by minimizing its expected cost. On top of that, a contrastive learning stage is further devised to utilize those pixels with unreliable predictions for a more compact target feature distribution. Extensive experiments on a cross-modality medical segmentation task demonstrate the superiority of our method in large domain discrepancy settings compared with the state-of-the-art SFDA approaches and even some UDA methods. Code is available at https://github.com/CSCYQJ/MICCAI23-ProtoContra-SFDA.

{{</citation>}}


### (26/106) Towards Building More Robust Models with Frequency Bias (Qingwen Bu et al., 2023)

{{<citation>}}

Qingwen Bu, Dong Huang, Heming Cui. (2023)  
**Towards Building More Robust Models with Frequency Bias**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.09763v1)  

---


**ABSTRACT**  
The vulnerability of deep neural networks to adversarial samples has been a major impediment to their broad applications, despite their success in various fields. Recently, some works suggested that adversarially-trained models emphasize the importance of low-frequency information to achieve higher robustness. While several attempts have been made to leverage this frequency characteristic, they have all faced the issue that applying low-pass filters directly to input images leads to irreversible loss of discriminative information and poor generalizability to datasets with distinct frequency features. This paper presents a plug-and-play module called the Frequency Preference Control Module that adaptively reconfigures the low- and high-frequency components of intermediate feature representations, providing better utilization of frequency in robust learning. Empirical studies show that our proposed module can be easily incorporated into any adversarial training framework, further improving model robustness across different architectures and datasets. Additionally, experiments were conducted to examine how the frequency bias of robust models impacts the adversarial training process and its final robustness, revealing interesting insights.

{{</citation>}}


### (27/106) Longitudinal Data and a Semantic Similarity Reward for Chest X-Ray Report Generation (Aaron Nicolson et al., 2023)

{{<citation>}}

Aaron Nicolson, Jason Dowling, Bevan Koopman. (2023)  
**Longitudinal Data and a Semantic Similarity Reward for Chest X-Ray Report Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT, Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2307.09758v1)  

---


**ABSTRACT**  
Chest X-Ray (CXR) report generation is a promising approach to improving the efficiency of CXR interpretation. However, a significant increase in diagnostic accuracy is required before that can be realised. Motivated by this, we propose a framework that is more inline with a radiologist's workflow by considering longitudinal data. Here, the decoder is additionally conditioned on the report from the subject's previous imaging study via a prompt. We also propose a new reward for reinforcement learning based on CXR-BERT, which computes the similarity between reports. We conduct experiments on the MIMIC-CXR dataset. The results indicate that longitudinal data improves CXR report generation. CXR-BERT is also shown to be a promising alternative to the current state-of-the-art reward based on RadGraph. This investigation indicates that longitudinal CXR report generation can offer a substantial increase in diagnostic accuracy. Our Hugging Face model is available at: https://huggingface.co/aehrc/cxrmate and code is available at: https://github.com/aehrc/cxrmate.

{{</citation>}}


### (28/106) Space Engage: Collaborative Space Supervision for Contrastive-based Semi-Supervised Semantic Segmentation (Changqi Wang et al., 2023)

{{<citation>}}

Changqi Wang, Haoyu Xie, Yuhui Yuan, Chong Fu, Xiangyu Yue. (2023)  
**Space Engage: Collaborative Space Supervision for Contrastive-based Semi-Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.09755v1)  

---


**ABSTRACT**  
Semi-Supervised Semantic Segmentation (S4) aims to train a segmentation model with limited labeled images and a substantial volume of unlabeled images. To improve the robustness of representations, powerful methods introduce a pixel-wise contrastive learning approach in latent space (i.e., representation space) that aggregates the representations to their prototypes in a fully supervised manner. However, previous contrastive-based S4 methods merely rely on the supervision from the model's output (logits) in logit space during unlabeled training. In contrast, we utilize the outputs in both logit space and representation space to obtain supervision in a collaborative way. The supervision from two spaces plays two roles: 1) reduces the risk of over-fitting to incorrect semantic information in logits with the help of representations; 2) enhances the knowledge exchange between the two spaces. Furthermore, unlike previous approaches, we use the similarity between representations and prototypes as a new indicator to tilt training those under-performing representations and achieve a more efficient contrastive learning process. Results on two public benchmarks demonstrate the competitive performance of our method compared with state-of-the-art methods.

{{</citation>}}


### (29/106) CPCM: Contextual Point Cloud Modeling for Weakly-supervised Point Cloud Semantic Segmentation (Lizhao Liu et al., 2023)

{{<citation>}}

Lizhao Liu, Zhuangwei Zhuang, Shangxin Huang, Xunlong Xiao, Tianhang Xiang, Cen Chen, Jingdong Wang, Mingkui Tan. (2023)  
**CPCM: Contextual Point Cloud Modeling for Weakly-supervised Point Cloud Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.10316v1)  

---


**ABSTRACT**  
We study the task of weakly-supervised point cloud semantic segmentation with sparse annotations (e.g., less than 0.1% points are labeled), aiming to reduce the expensive cost of dense annotations. Unfortunately, with extremely sparse annotated points, it is very difficult to extract both contextual and object information for scene understanding such as semantic segmentation. Motivated by masked modeling (e.g., MAE) in image and video representation learning, we seek to endow the power of masked modeling to learn contextual information from sparsely-annotated points. However, directly applying MAE to 3D point clouds with sparse annotations may fail to work. First, it is nontrivial to effectively mask out the informative visual context from 3D point clouds. Second, how to fully exploit the sparse annotations for context modeling remains an open question. In this paper, we propose a simple yet effective Contextual Point Cloud Modeling (CPCM) method that consists of two parts: a region-wise masking (RegionMask) strategy and a contextual masked training (CMT) method. Specifically, RegionMask masks the point cloud continuously in geometric space to construct a meaningful masked prediction task for subsequent context learning. CMT disentangles the learning of supervised segmentation and unsupervised masked context prediction for effectively learning the very limited labeled points and mass unlabeled points, respectively. Extensive experiments on the widely-tested ScanNet V2 and S3DIS benchmarks demonstrate the superiority of CPCM over the state-of-the-art.

{{</citation>}}


### (30/106) NTIRE 2023 Quality Assessment of Video Enhancement Challenge (Xiaohong Liu et al., 2023)

{{<citation>}}

Xiaohong Liu, Xiongkuo Min, Wei Sun, Yulun Zhang, Kai Zhang, Radu Timofte, Guangtao Zhai, Yixuan Gao, Yuqin Cao, Tengchuan Kou, Yunlong Dong, Ziheng Jia, Yilin Li, Wei Wu, Shuming Hu, Sibin Deng, Pengxiang Xiao, Ying Chen, Kai Li, Kai Zhao, Kun Yuan, Ming Sun, Heng Cong, Hao Wang, Lingzhi Fu, Yusheng Zhang, Rongyu Zhang, Hang Shi, Qihang Xu, Longan Xiao, Zhiliang Ma, Mirko Agarla, Luigi Celona, Claudio Rota, Raimondo Schettini, Zhiwei Huang, Yanan Li, Xiaotao Wang, Lei Lei, Hongye Liu, Wei Hong, Ironhead Chuang, Allen Lin, Drake Guan, Iris Chen, Kae Lou, Willy Huang, Yachun Tasi, Yvonne Kao, Haotian Fan, Fangyuan Kong, Shiqi Zhou, Hao Liu, Yu Lai, Shanshan Chen, Wenqi Wang, Haoning Wu, Chaofeng Chen, Chunzheng Zhu, Zekun Guo, Shiling Zhao, Haibing Yin, Hongkui Wang, Hanene Brachemi Meftah, Sid Ahmed Fezza, Wassim Hamidouche, Olivier Déforges, Tengfei Shi, Azadeh Mansouri, Hossein Motamednia, Amir Hossein Bakhtiari, Ahmad Mahmoudi Aznaveh. (2023)  
**NTIRE 2023 Quality Assessment of Video Enhancement Challenge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.09729v1)  

---


**ABSTRACT**  
This paper reports on the NTIRE 2023 Quality Assessment of Video Enhancement Challenge, which will be held in conjunction with the New Trends in Image Restoration and Enhancement Workshop (NTIRE) at CVPR 2023. This challenge is to address a major challenge in the field of video processing, namely, video quality assessment (VQA) for enhanced videos. The challenge uses the VQA Dataset for Perceptual Video Enhancement (VDPVE), which has a total of 1211 enhanced videos, including 600 videos with color, brightness, and contrast enhancements, 310 videos with deblurring, and 301 deshaked videos. The challenge has a total of 167 registered participants. 61 participating teams submitted their prediction results during the development phase, with a total of 3168 submissions. A total of 176 submissions were submitted by 37 participating teams during the final testing phase. Finally, 19 participating teams submitted their models and fact sheets, and detailed the methods they used. Some methods have achieved better results than baseline methods, and the winning methods have demonstrated superior prediction performance.

{{</citation>}}


### (31/106) SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid (Zi Li et al., 2023)

{{<citation>}}

Zi Li, Lin Tian, Tony C. W. Mok, Xiaoyu Bai, Puyang Wang, Jia Ge, Jingren Zhou, Le Lu, Xianghua Ye, Ke Yan, Dakai Jin. (2023)  
**SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.09727v1)  

---


**ABSTRACT**  
Estimating displacement vector field via a cost volume computed in the feature space has shown great success in image registration, but it suffers excessive computation burdens. Moreover, existing feature descriptors only extract local features incapable of representing the global semantic information, which is especially important for solving large transformations. To address the discussed issues, we propose SAMConvex, a fast coarse-to-fine discrete optimization method for CT registration that includes a decoupled convex optimization procedure to obtain deformation fields based on a self-supervised anatomical embedding (SAM) feature extractor that captures both local and global information. To be specific, SAMConvex extracts per-voxel features and builds 6D correlation volumes based on SAM features, and iteratively updates a flow field by performing lookups on the correlation volumes with a coarse-to-fine scheme. SAMConvex outperforms the state-of-the-art learning-based methods and optimization-based methods over two inter-patient registration datasets (Abdomen CT and HeadNeck CT) and one intra-patient registration dataset (Lung CT). Moreover, as an optimization-based method, SAMConvex only takes $\sim2$s ($\sim5s$ with instance optimization) for one paired images.

{{</citation>}}


### (32/106) AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks (Kibeom Hong et al., 2023)

{{<citation>}}

Kibeom Hong, Seogkyu Jeon, Junsoo Lee, Namhyuk Ahn, Kunhee Kim, Pilhyeon Lee, Daesik Kim, Youngjung Uh, Hyeran Byun. (2023)  
**AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2307.09724v2)  

---


**ABSTRACT**  
To deliver the artistic expression of the target style, recent studies exploit the attention mechanism owing to its ability to map the local patches of the style image to the corresponding patches of the content image. However, because of the low semantic correspondence between arbitrary content and artworks, the attention module repeatedly abuses specific local patches from the style image, resulting in disharmonious and evident repetitive artifacts. To overcome this limitation and accomplish impeccable artistic style transfer, we focus on enhancing the attention mechanism and capturing the rhythm of patterns that organize the style. In this paper, we introduce a novel metric, namely pattern repeatability, that quantifies the repetition of patterns in the style image. Based on the pattern repeatability, we propose Aesthetic Pattern-Aware style transfer Networks (AesPA-Net) that discover the sweet spot of local and global style expressions. In addition, we propose a novel self-supervisory task to encourage the attention mechanism to learn precise and meaningful semantic correspondence. Lastly, we introduce the patch-wise style loss to transfer the elaborate rhythm of local patterns. Through qualitative and quantitative evaluations, we verify the reliability of the proposed pattern repeatability that aligns with human perception, and demonstrate the superiority of the proposed framework.

{{</citation>}}


### (33/106) Semantic-Aware Dual Contrastive Learning for Multi-label Image Classification (Leilei Ma et al., 2023)

{{<citation>}}

Leilei Ma, Dengdi Sun, Lei Wang, Haifang Zhao, Bin Luo. (2023)  
**Semantic-Aware Dual Contrastive Learning for Multi-label Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Image Classification  
[Paper Link](http://arxiv.org/abs/2307.09715v1)  

---


**ABSTRACT**  
Extracting image semantics effectively and assigning corresponding labels to multiple objects or attributes for natural images is challenging due to the complex scene contents and confusing label dependencies. Recent works have focused on modeling label relationships with graph and understanding object regions using class activation maps (CAM). However, these methods ignore the complex intra- and inter-category relationships among specific semantic features, and CAM is prone to generate noisy information. To this end, we propose a novel semantic-aware dual contrastive learning framework that incorporates sample-to-sample contrastive learning (SSCL) as well as prototype-to-sample contrastive learning (PSCL). Specifically, we leverage semantic-aware representation learning to extract category-related local discriminative features and construct category prototypes. Then based on SSCL, label-level visual representations of the same category are aggregated together, and features belonging to distinct categories are separated. Meanwhile, we construct a novel PSCL module to narrow the distance between positive samples and category prototypes and push negative samples away from the corresponding category prototypes. Finally, the discriminative label-level features related to the image content are accurately captured by the joint training of the above three parts. Experiments on five challenging large-scale public datasets demonstrate that our proposed method is effective and outperforms the state-of-the-art methods. Code and supplementary materials are released on https://github.com/yu-gi-oh-leilei/SADCL.

{{</citation>}}


## cs.LG (13)



### (34/106) Novel Batch Active Learning Approach and Its Application to Synthetic Aperture Radar Datasets (James Chapman et al., 2023)

{{<citation>}}

James Chapman, Bohan Chen, Zheng Tan, Jeff Calder, Kevin Miller, Andrea L. Bertozzi. (2023)  
**Novel Batch Active Learning Approach and Its Application to Synthetic Aperture Radar Datasets**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-10; I-4-0; I-4-9, cs-CV, cs-LG, cs.LG, eess-SP  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.10495v1)  

---


**ABSTRACT**  
Active learning improves the performance of machine learning methods by judiciously selecting a limited number of unlabeled data points to query for labels, with the aim of maximally improving the underlying classifier's performance. Recent gains have been made using sequential active learning for synthetic aperture radar (SAR) data arXiv:2204.00005. In each iteration, sequential active learning selects a query set of size one while batch active learning selects a query set of multiple datapoints. While batch active learning methods exhibit greater efficiency, the challenge lies in maintaining model accuracy relative to sequential active learning methods. We developed a novel, two-part approach for batch active learning: Dijkstra's Annulus Core-Set (DAC) for core-set generation and LocalMax for batch sampling. The batch active learning process that combines DAC and LocalMax achieves nearly identical accuracy as sequential active learning but is more efficient, proportional to the batch size. As an application, a pipeline is built based on transfer learning feature embedding, graph learning, DAC, and LocalMax to classify the FUSAR-Ship and OpenSARShip datasets. Our pipeline outperforms the state-of-the-art CNN-based methods.

{{</citation>}}


### (35/106) Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search (Shengli Jiang et al., 2023)

{{<citation>}}

Shengli Jiang, Shiyi Qin, Reid C. Van Lehn, Prasanna Balaprakash, Victor M. Zavala. (2023)  
**Uncertainty Quantification for Molecular Property Predictions with Graph Neural Architecture Search**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-chem-ph, q-bio-BM  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.10438v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have emerged as a prominent class of data-driven methods for molecular property prediction. However, a key limitation of typical GNN models is their inability to quantify uncertainties in the predictions. This capability is crucial for ensuring the trustworthy use and deployment of models in downstream tasks. To that end, we introduce AutoGNNUQ, an automated uncertainty quantification (UQ) approach for molecular property prediction. AutoGNNUQ leverages architecture search to generate an ensemble of high-performing GNNs, enabling the estimation of predictive uncertainties. Our approach employs variance decomposition to separate data (aleatoric) and model (epistemic) uncertainties, providing valuable insights for reducing them. In our computational experiments, we demonstrate that AutoGNNUQ outperforms existing UQ methods in terms of both prediction accuracy and UQ performance on multiple benchmark datasets. Additionally, we utilize t-SNE visualization to explore correlations between molecular features and uncertainty, offering insight for dataset improvement. AutoGNNUQ has broad applicability in domains such as drug discovery and materials science, where accurate uncertainty quantification is crucial for decision-making.

{{</citation>}}


### (36/106) DP-TBART: A Transformer-based Autoregressive Model for Differentially Private Tabular Data Generation (Rodrigo Castellon et al., 2023)

{{<citation>}}

Rodrigo Castellon, Achintya Gopal, Brian Bloniarz, David Rosenberg. (2023)  
**DP-TBART: A Transformer-based Autoregressive Model for Differentially Private Tabular Data Generation**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.10430v1)  

---


**ABSTRACT**  
The generation of synthetic tabular data that preserves differential privacy is a problem of growing importance. While traditional marginal-based methods have achieved impressive results, recent work has shown that deep learning-based approaches tend to lag behind. In this work, we present Differentially-Private TaBular AutoRegressive Transformer (DP-TBART), a transformer-based autoregressive model that maintains differential privacy and achieves performance competitive with marginal-based methods on a wide variety of datasets, capable of even outperforming state-of-the-art methods in certain settings. We also provide a theoretical framework for understanding the limitations of marginal-based approaches and where deep learning-based approaches stand to contribute most. These results suggest that deep learning-based techniques should be considered as a viable alternative to marginal-based methods in the generation of differentially private synthetic tabular data.

{{</citation>}}


### (37/106) LightPath: Lightweight and Scalable Path Representation Learning (Sean Bin Yang et al., 2023)

{{<citation>}}

Sean Bin Yang, Jilin Hu, Chenjuan Guo, Bin Yang, Christian S. Jensen. (2023)  
**LightPath: Lightweight and Scalable Path Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DB, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.10171v1)  

---


**ABSTRACT**  
Movement paths are used widely in intelligent transportation and smart city applications. To serve such applications, path representation learning aims to provide compact representations of paths that enable efficient and accurate operations when used for different downstream tasks such as path ranking and travel cost estimation. In many cases, it is attractive that the path representation learning is lightweight and scalable; in resource-limited environments and under green computing limitations, it is essential. Yet, existing path representation learning studies focus on accuracy and pay at most secondary attention to resource consumption and scalability.   We propose a lightweight and scalable path representation learning framework, termed LightPath, that aims to reduce resource consumption and achieve scalability without affecting accuracy, thus enabling broader applicability. More specifically, we first propose a sparse auto-encoder that ensures that the framework achieves good scalability with respect to path length. Next, we propose a relational reasoning framework to enable faster training of more robust sparse path encoders. We also propose global-local knowledge distillation to further reduce the size and improve the performance of sparse path encoders. Finally, we report extensive experiments on two real-world datasets to offer insight into the efficiency, scalability, and effectiveness of the proposed framework.

{{</citation>}}


### (38/106) Improving Multimodal Datasets with Image Captioning (Thao Nguyen et al., 2023)

{{<citation>}}

Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, Ludwig Schmidt. (2023)  
**Improving Multimodal Datasets with Image Captioning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Image Captioning, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.10350v1)  

---


**ABSTRACT**  
Massive web datasets play a key role in the success of large vision-language models like CLIP and Flamingo. However, the raw web data is noisy, and existing filtering methods to reduce noise often come at the expense of data diversity. Our work focuses on caption quality as one major source of noise, and studies how generated captions can increase the utility of web-scraped datapoints with nondescript text. Through exploring different mixing strategies for raw and generated captions, we outperform the best filtering method proposed by the DataComp benchmark by 2% on ImageNet and 4% on average across 38 tasks, given a candidate pool of 128M image-text pairs. Our best approach is also 2x better at Flickr and MS-COCO retrieval. We then analyze what makes synthetic captions an effective source of text supervision. In experimenting with different image captioning models, we also demonstrate that the performance of a model on standard image captioning benchmarks (e.g., NoCaps CIDEr) is not a reliable indicator of the utility of the captions it generates for multimodal training. Finally, our experiments with using generated captions at DataComp's large scale (1.28B image-text pairs) offer insights into the limitations of synthetic text, as well as the importance of image curation with increasing training data quantity.

{{</citation>}}


### (39/106) Android in the Wild: A Large-Scale Dataset for Android Device Control (Christopher Rawles et al., 2023)

{{<citation>}}

Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, Timothy Lillicrap. (2023)  
**Android in the Wild: A Large-Scale Dataset for Android Device Control**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10088v1)  

---


**ABSTRACT**  
There is a growing interest in device-control systems that can interpret human natural language instructions and execute them on a digital device by directly controlling its user interface. We present a dataset for device-control research, Android in the Wild (AITW), which is orders of magnitude larger than current datasets. The dataset contains human demonstrations of device interactions, including the screens and actions, and corresponding natural language instructions. It consists of 715k episodes spanning 30k unique instructions, four versions of Android (v10-13),and eight device types (Pixel 2 XL to Pixel 6) with varying screen resolutions. It contains multi-step tasks that require semantic understanding of language and visual context. This dataset poses a new challenge: actions available through the user interface must be inferred from their visual appearance. And, instead of simple UI element-based actions, the action space consists of precise gestures (e.g., horizontal scrolls to operate carousel widgets). We organize our dataset to encourage robustness analysis of device-control systems, i.e., how well a system performs in the presence of new task descriptions, new applications, or new platform versions. We develop two agents and report performance across the dataset. The dataset is available at https://github.com/google-research/google-research/tree/master/android_in_the_wild.

{{</citation>}}


### (40/106) Impact of Disentanglement on Pruning Neural Networks (Carl Shneider et al., 2023)

{{<citation>}}

Carl Shneider, Peyman Rostami, Anis Kacem, Nilotpal Sinha, Abd El Rahman Shabayek, Djamila Aouada. (2023)  
**Impact of Disentanglement on Pruning Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, eess-SP  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2307.09994v1)  

---


**ABSTRACT**  
Deploying deep learning neural networks on edge devices, to accomplish task specific objectives in the real-world, requires a reduction in their memory footprint, power consumption, and latency. This can be realized via efficient model compression. Disentangled latent representations produced by variational autoencoder (VAE) networks are a promising approach for achieving model compression because they mainly retain task-specific information, discarding useless information for the task at hand. We make use of the Beta-VAE framework combined with a standard criterion for pruning to investigate the impact of forcing the network to learn disentangled representations on the pruning process for the task of classification. In particular, we perform experiments on MNIST and CIFAR10 datasets, examine disentanglement challenges, and propose a path forward for future works.

{{</citation>}}


### (41/106) TREEMENT: Interpretable Patient-Trial Matching via Personalized Dynamic Tree-Based Memory Network (Brandon Theodorou et al., 2023)

{{<citation>}}

Brandon Theodorou, Cao Xiao, Jimeng Sun. (2023)  
**TREEMENT: Interpretable Patient-Trial Matching via Personalized Dynamic Tree-Based Memory Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2307.09942v1)  

---


**ABSTRACT**  
Clinical trials are critical for drug development but often suffer from expensive and inefficient patient recruitment. In recent years, machine learning models have been proposed for speeding up patient recruitment via automatically matching patients with clinical trials based on longitudinal patient electronic health records (EHR) data and eligibility criteria of clinical trials. However, they either depend on trial-specific expert rules that cannot expand to other trials or perform matching at a very general level with a black-box model where the lack of interpretability makes the model results difficult to be adopted.   To provide accurate and interpretable patient trial matching, we introduce a personalized dynamic tree-based memory network model named TREEMENT. It utilizes hierarchical clinical ontologies to expand the personalized patient representation learned from sequential EHR data, and then uses an attentional beam-search query learned from eligibility criteria embedding to offer a granular level of alignment for improved performance and interpretability. We evaluated TREEMENT against existing models on real-world datasets and demonstrated that TREEMENT outperforms the best baseline by 7% in terms of error reduction in criteria-level matching and achieves state-of-the-art results in its trial-level matching ability. Furthermore, we also show TREEMENT can offer good interpretability to make the model results easier for adoption.

{{</citation>}}


### (42/106) Reproducibility in Machine Learning-Driven Research (Harald Semmelrock et al., 2023)

{{<citation>}}

Harald Semmelrock, Simone Kopeinik, Dieter Theiler, Tony Ross-Hellauer, Dominik Kowald. (2023)  
**Reproducibility in Machine Learning-Driven Research**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG, stat-ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10320v1)  

---


**ABSTRACT**  
Research is facing a reproducibility crisis, in which the results and findings of many studies are difficult or even impossible to reproduce. This is also the case in machine learning (ML) and artificial intelligence (AI) research. Often, this is the case due to unpublished data and/or source-code, and due to sensitivity to ML training conditions. Although different solutions to address this issue are discussed in the research community such as using ML platforms, the level of reproducibility in ML-driven research is not increasing substantially. Therefore, in this mini survey, we review the literature on reproducibility in ML-driven research with three main aims: (i) reflect on the current situation of ML reproducibility in various research fields, (ii) identify reproducibility issues and barriers that exist in these research fields applying ML, and (iii) identify potential drivers such as tools, practices, and interventions that support ML reproducibility. With this, we hope to contribute to decisions on the viability of different solutions for supporting ML reproducibility.

{{</citation>}}


### (43/106) ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats (Xiaoxia Wu et al., 2023)

{{<citation>}}

Xiaoxia Wu, Zhewei Yao, Yuxiong He. (2023)  
**ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2307.09782v2)  

---


**ABSTRACT**  
In the complex domain of large language models (LLMs), striking a balance between computational efficiency and maintaining model quality is a formidable challenge. Navigating the inherent limitations of uniform quantization, particularly when dealing with outliers, and motivated by the launch of NVIDIA's H100 hardware, this study delves into the viability of floating-point (FP) quantization, particularly focusing on FP8 and FP4, as a potential solution. Our comprehensive investigation reveals that for LLMs, FP8 activation consistently outshines its integer (INT8) equivalent, with the performance edge becoming more noticeable in models possessing parameters beyond one billion. For weight quantization, our findings indicate that FP4 exhibits comparable, if not superior, performance to INT4, simplifying deployment on FP-supported hardware like H100. To mitigate the overhead from precision alignment caused by the disparity between weights and activations, we propose two scaling constraints for weight quantization that negligibly impact the performance compared to the standard W4A8 model. We additionally enhance our quantization methods by integrating the Low Rank Compensation (LoRC) strategy, yielding improvements especially in smaller models. The results of our investigation emphasize the immense potential of FP quantization for LLMs, paving the way for high-efficiency deployment in resource-limited settings.

{{</citation>}}


### (44/106) How Curvature Enhance the Adaptation Power of Framelet GCNs (Dai Shi et al., 2023)

{{<citation>}}

Dai Shi, Yi Guo, Zhiqi Shao, Junbin Gao. (2023)  
**How Curvature Enhance the Adaptation Power of Framelet GCNs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2307.09768v1)  

---


**ABSTRACT**  
Graph neural network (GNN) has been demonstrated powerful in modeling graph-structured data. However, despite many successful cases of applying GNNs to various graph classification and prediction tasks, whether the graph geometrical information has been fully exploited to enhance the learning performance of GNNs is not yet well understood. This paper introduces a new approach to enhance GNN by discrete graph Ricci curvature. Specifically, the graph Ricci curvature defined on the edges of a graph measures how difficult the information transits on one edge from one node to another based on their neighborhoods. Motivated by the geometric analogy of Ricci curvature in the graph setting, we prove that by inserting the curvature information with different carefully designed transformation function $\zeta$, several known computational issues in GNN such as over-smoothing can be alleviated in our proposed model. Furthermore, we verified that edges with very positive Ricci curvature (i.e., $\kappa_{i,j} \approx 1$) are preferred to be dropped to enhance model's adaption to heterophily graph and one curvature based graph edge drop algorithm is proposed. Comprehensive experiments show that our curvature-based GNN model outperforms the state-of-the-art baselines in both homophily and heterophily graph datasets, indicating the effectiveness of involving graph geometric information in GNNs.

{{</citation>}}


### (45/106) Constructing Extreme Learning Machines with zero Spectral Bias (Kaumudi Joshi et al., 2023)

{{<citation>}}

Kaumudi Joshi, Vukka Snigdha, Arya Kumar Bhattacharya. (2023)  
**Constructing Extreme Learning Machines with zero Spectral Bias**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias, Embedding  
[Paper Link](http://arxiv.org/abs/2307.09759v1)  

---


**ABSTRACT**  
The phenomena of Spectral Bias, where the higher frequency components of a function being learnt in a feedforward Artificial Neural Network (ANN) are seen to converge more slowly than the lower frequencies, is observed ubiquitously across ANNs. This has created technology challenges in fields where resolution of higher frequencies is crucial, like in Physics Informed Neural Networks (PINNs). Extreme Learning Machines (ELMs) that obviate an iterative solution process which provides the theoretical basis of Spectral Bias (SB), should in principle be free of the same. This work verifies the reliability of this assumption, and shows that it is incorrect. However, the structure of ELMs makes them naturally amenable to implementation of variants of Fourier Feature Embeddings, which have been shown to mitigate SB in ANNs. This approach is implemented and verified to completely eliminate SB, thus bringing into feasibility the application of ELMs for practical problems like PINNs where resolution of higher frequencies is essential.

{{</citation>}}


### (46/106) STRAPPER: Preference-based Reinforcement Learning via Self-training Augmentation and Peer Regularization (Yachen Kang et al., 2023)

{{<citation>}}

Yachen Kang, Li He, Jinxin Liu, Zifeng Zhuang, Donglin Wang. (2023)  
**STRAPPER: Preference-based Reinforcement Learning via Self-training Augmentation and Peer Regularization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09692v1)  

---


**ABSTRACT**  
Preference-based reinforcement learning (PbRL) promises to learn a complex reward function with binary human preference. However, such human-in-the-loop formulation requires considerable human effort to assign preference labels to segment pairs, hindering its large-scale applications. Recent approache has tried to reuse unlabeled segments, which implicitly elucidates the distribution of segments and thereby alleviates the human effort. And consistency regularization is further considered to improve the performance of semi-supervised learning. However, we notice that, unlike general classification tasks, in PbRL there exits a unique phenomenon that we defined as similarity trap in this paper. Intuitively, human can have diametrically opposite preferredness for similar segment pairs, but such similarity may trap consistency regularization fail in PbRL. Due to the existence of similarity trap, such consistency regularization improperly enhances the consistency possiblity of the model's predictions between segment pairs, and thus reduces the confidence in reward learning, since the augmented distribution does not match with the original one in PbRL. To overcome such issue, we present a self-training method along with our proposed peer regularization, which penalizes the reward model memorizing uninformative labels and acquires confident predictions. Empirically, we demonstrate that our approach is capable of learning well a variety of locomotion and robotic manipulation behaviors using different semi-supervised alternatives and peer regularization.

{{</citation>}}


## cs.CR (3)



### (47/106) (Ab)using Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs (Eugene Bagdasaryan et al., 2023)

{{<citation>}}

Eugene Bagdasaryan, Tsung-Yin Hsieh, Ben Nassi, Vitaly Shmatikov. (2023)  
**(Ab)using Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.10490v2)  

---


**ABSTRACT**  
We demonstrate how images and sounds can be used for indirect prompt and instruction injection in multi-modal LLMs. An attacker generates an adversarial perturbation corresponding to the prompt and blends it into an image or audio recording. When the user asks the (unmodified, benign) model about the perturbed image or audio, the perturbation steers the model to output the attacker-chosen text and/or make the subsequent dialog follow the attacker's instruction. We illustrate this attack with several proof-of-concept examples targeting LLaVa and PandaGPT.

{{</citation>}}


### (48/106) What can we learn from Data Leakage and Unlearning for Law? (Jaydeep Borkar, 2023)

{{<citation>}}

Jaydeep Borkar. (2023)  
**What can we learn from Data Leakage and Unlearning for Law?**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.10476v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have a privacy concern because they memorize training data (including personally identifiable information (PII) like emails and phone numbers) and leak it during inference. A company can train an LLM on its domain-customized data which can potentially also include their users' PII. In order to comply with privacy laws such as the "right to be forgotten", the data points of users that are most vulnerable to extraction could be deleted. We find that once the most vulnerable points are deleted, a new set of points become vulnerable to extraction. So far, little attention has been given to understanding memorization for fine-tuned models. In this work, we also show that not only do fine-tuned models leak their training data but they also leak the pre-training data (and PII) memorized during the pre-training phase. The property of new data points becoming vulnerable to extraction after unlearning and leakage of pre-training data through fine-tuned models can pose significant privacy and legal concerns for companies that use LLMs to offer services. We hope this work will start an interdisciplinary discussion within AI and law communities regarding the need for policies to tackle these issues.

{{</citation>}}


### (49/106) NFT-Based Blockchain-Oriented Security Framework for Metaverse Applications (Khadija Manzoor et al., 2023)

{{<citation>}}

Khadija Manzoor, Umara Noor, Zahid Rashid. (2023)  
**NFT-Based Blockchain-Oriented Security Framework for Metaverse Applications**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.10342v1)  

---


**ABSTRACT**  
The Metaverse is rapidly evolving, bringing us closer to its imminent reality. However, the widespread adoption of this new automated technology poses significant research challenges in terms of authenticity, integrity, interoperability, and efficiency. These challenges originate from the core technologies underlying the Metaverse and are exacerbated by its complex nature. As a solution to these challenges, this paper presents a novel framework based on Non-Fungible Tokens (NFTs). The framework employs the Proof-of-Stake consensus algorithm, a blockchain-based technology, for data transaction, validation, and resource management. PoS efficiently consume energy and provide a streamlined validation approach instead of resource-intensive mining. This ability makes PoS an ideal candidate for Metaverse applications. By combining NFTs for user authentication and PoS for data integrity, enhanced transaction throughput, and improved scalability, the proposed blockchain mechanism demonstrates noteworthy advantages. Through security analysis, experimental and simulation results, it is established that the NFT-based approach coupled with the PoS algorithm is secure and efficient for Metaverse applications.

{{</citation>}}


## cs.IR (6)



### (50/106) SPRINT: A Unified Toolkit for Evaluating and Demystifying Zero-shot Neural Sparse Retrieval (Nandan Thakur et al., 2023)

{{<citation>}}

Nandan Thakur, Kexin Wang, Iryna Gurevych, Jimmy Lin. (2023)  
**SPRINT: A Unified Toolkit for Evaluating and Demystifying Zero-shot Neural Sparse Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.10488v1)  

---


**ABSTRACT**  
Traditionally, sparse retrieval systems relied on lexical representations to retrieve documents, such as BM25, dominated information retrieval tasks. With the onset of pre-trained transformer models such as BERT, neural sparse retrieval has led to a new paradigm within retrieval. Despite the success, there has been limited software supporting different sparse retrievers running in a unified, common environment. This hinders practitioners from fairly comparing different sparse models and obtaining realistic evaluation results. Another missing piece is, that a majority of prior work evaluates sparse retrieval models on in-domain retrieval, i.e. on a single dataset: MS MARCO. However, a key requirement in practical retrieval systems requires models that can generalize well to unseen out-of-domain, i.e. zero-shot retrieval tasks. In this work, we provide SPRINT, a unified Python toolkit based on Pyserini and Lucene, supporting a common interface for evaluating neural sparse retrieval. The toolkit currently includes five built-in models: uniCOIL, DeepImpact, SPARTA, TILDEv2 and SPLADEv2. Users can also easily add customized models by defining their term weighting method. Using our toolkit, we establish strong and reproducible zero-shot sparse retrieval baselines across the well-acknowledged benchmark, BEIR. Our results demonstrate that SPLADEv2 achieves the best average score of 0.470 nDCG@10 on BEIR amongst all neural sparse retrievers. In this work, we further uncover the reasons behind its performance gain. We show that SPLADEv2 produces sparse representations with a majority of tokens outside of the original query and document which is often crucial for its performance gains, i.e. a limitation among its other sparse counterparts. We provide our SPRINT toolkit, models, and data used in our experiments publicly here at https://github.com/thakur-nandan/sprint.

{{</citation>}}


### (51/106) Who Provides the Largest Megaphone? The Role of Google News in Promoting Russian State-Affiliated News Sources (Keeley Erhardt et al., 2023)

{{<citation>}}

Keeley Erhardt, Saurabh Khanna. (2023)  
**Who Provides the Largest Megaphone? The Role of Google News in Promoting Russian State-Affiliated News Sources**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2307.09834v1)  

---


**ABSTRACT**  
The Internet has not only digitized but also democratized information access across the globe. This gradual but path-breaking move to online information propagation has resulted in search engines playing an increasingly prominent role in shaping access to human knowledge. When an Internet user enters a query, the search engine sorts through the hundreds of billions of possible webpages to determine what to show. Google dominates the search engine market, with Google Search surpassing 80% market share globally every year of the last decade. Only in Russia and China do Google competitors claim more market share, with approximately 60% of Internet users in Russia preferring Yandex (compared to 40% in favor of Google) and more than 80% of China's Internet users accessing Baidu as of 2022. Notwithstanding this long-standing regional variation in Internet search providers, there is limited research showing how these providers compare in terms of propagating state-sponsored information. Our study fills this research gap by focusing on Russian cyberspace and examining how Google and Yandex's search algorithms rank content from Russian state-controlled media (hereon, RSM) outlets. This question is timely and of practical interest given widespread reports indicating that RSM outlets have actively engaged in promoting Kremlin propaganda in the lead-up to, and in the aftermath of, the Russian invasion of Ukraine in February 2022.

{{</citation>}}


### (52/106) DisCover: Disentangled Music Representation Learning for Cover Song Identification (Jiahao Xun et al., 2023)

{{<citation>}}

Jiahao Xun, Shengyu Zhang, Yanting Yang, Jieming Zhu, Liqun Deng, Zhou Zhao, Zhenhua Dong, Ruiqi Li, Lichao Zhang, Fei Wu. (2023)  
**DisCover: Disentangled Music Representation Learning for Cover Song Identification**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-SD, cs.IR, eess-AS  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.09775v1)  

---


**ABSTRACT**  
In the field of music information retrieval (MIR), cover song identification (CSI) is a challenging task that aims to identify cover versions of a query song from a massive collection. Existing works still suffer from high intra-song variances and inter-song correlations, due to the entangled nature of version-specific and version-invariant factors in their modeling. In this work, we set the goal of disentangling version-specific and version-invariant factors, which could make it easier for the model to learn invariant music representations for unseen query songs. We analyze the CSI task in a disentanglement view with the causal graph technique, and identify the intra-version and inter-version effects biasing the invariant learning. To block these effects, we propose the disentangled music representation learning framework (DisCover) for CSI. DisCover consists of two critical components: (1) Knowledge-guided Disentanglement Module (KDM) and (2) Gradient-based Adversarial Disentanglement Module (GADM), which block intra-version and inter-version biased effects, respectively. KDM minimizes the mutual information between the learned representations and version-variant factors that are identified with prior domain knowledge. GADM identifies version-variant factors by simulating the representation transitions between intra-song versions, and exploits adversarial distillation for effect blocking. Extensive comparisons with best-performing methods and in-depth analysis demonstrate the effectiveness of DisCover and the and necessity of disentanglement for CSI.

{{</citation>}}


### (53/106) Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community (Qingyao Ai et al., 2023)

{{<citation>}}

Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, Shen Gao, Jiafeng Guo, Xiangnan He, Yanyan Lan, Chenliang Li, Yiqun Liu, Ziyu Lyu, Weizhi Ma, Jun Ma, Zhaochun Ren, Pengjie Ren, Zhiqiang Wang, Mingwen Wang, Jirong Wen, Le Wu, Xin Xin, Jun Xu, Dawei Yin, Peng Zhang, Fan Zhang, Weinan Zhang, Min Zhang, Xiaofei Zhu. (2023)  
**Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2307.09751v1)  

---


**ABSTRACT**  
The research field of Information Retrieval (IR) has evolved significantly, expanding beyond traditional search to meet diverse user information needs. Recently, Large Language Models (LLMs) have demonstrated exceptional capabilities in text understanding, generation, and knowledge inference, opening up exciting avenues for IR research. LLMs not only facilitate generative retrieval but also offer improved solutions for user understanding, model evaluation, and user-system interactions. More importantly, the synergistic relationship among IR models, LLMs, and humans forms a new technical paradigm that is more powerful for information seeking. IR models provide real-time and relevant information, LLMs contribute internal knowledge, and humans play a central role of demanders and evaluators to the reliability of information services. Nevertheless, significant challenges exist, including computational costs, credibility concerns, domain-specific limitations, and ethical considerations. To thoroughly discuss the transformative impact of LLMs on IR research, the Chinese IR community conducted a strategic workshop in April 2023, yielding valuable insights. This paper provides a summary of the workshop's outcomes, including the rethinking of IR's core values, the mutual enhancement of LLMs and IR, the proposal of a novel IR technical paradigm, and open challenges.

{{</citation>}}


### (54/106) Mood Classification of Bangla Songs Based on Lyrics (Maliha Mahajebin et al., 2023)

{{<citation>}}

Maliha Mahajebin, Mohammad Rifat Ahmmad Rashid, Nafees Mansoor. (2023)  
**Mood Classification of Bangla Songs Based on Lyrics**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs-SD, cs.IR, eess-AS  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.10314v1)  

---


**ABSTRACT**  
Music can evoke various emotions, and with the advancement of technology, it has become more accessible to people. Bangla music, which portrays different human emotions, lacks sufficient research. The authors of this article aim to analyze Bangla songs and classify their moods based on the lyrics. To achieve this, this research has compiled a dataset of 4000 Bangla song lyrics, genres, and used Natural Language Processing and the Bert Algorithm to analyze the data. Among the 4000 songs, 1513 songs are represented for the sad mood, 1362 for the romantic mood, 886 for happiness, and the rest 239 are classified as relaxation. By embedding the lyrics of the songs, the authors have classified the songs into four moods: Happy, Sad, Romantic, and Relaxed. This research is crucial as it enables a multi-class classification of songs' moods, making the music more relatable to people's emotions. The article presents the automated result of the four moods accurately derived from the song lyrics.

{{</citation>}}


### (55/106) Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation (Wei Jin et al., 2023)

{{<citation>}}

Wei Jin, Haitao Mao, Zheng Li, Haoming Jiang, Chen Luo, Hongzhi Wen, Haoyu Han, Hanqing Lu, Zhengyang Wang, Ruirui Li, Zhen Li, Monica Xiao Cheng, Rahul Goutam, Haiyang Zhang, Karthik Subbian, Suhang Wang, Yizhou Sun, Jiliang Tang, Bing Yin, Xianfeng Tang. (2023)  
**Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Amazon, Multilingual, Text Generation  
[Paper Link](http://arxiv.org/abs/2307.09688v1)  

---


**ABSTRACT**  
Modeling customer shopping intentions is a crucial task for e-commerce, as it directly impacts user experience and engagement. Thus, accurately understanding customer preferences is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next interaction, has become increasingly popular. However, existing session datasets have limitations in terms of item attributes, user diversity, and dataset scale. As a result, they cannot comprehensively capture the spectrum of user behaviors and preferences. To bridge this gap, we present the Amazon Multilingual Multi-locale Shopping Session Dataset, namely Amazon-M2. It is the first multilingual dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish. Remarkably, the dataset can help us enhance personalization and understanding of user preferences, which can benefit various existing tasks as well as enable new tasks. To test the potential of the dataset, we introduce three tasks in this work: (1) next-product recommendation, (2) next-product recommendation with domain shifts, and (3) next-product title generation. With the above tasks, we benchmark a range of algorithms on our proposed dataset, drawing new insights for further research and practice. In addition, based on the proposed dataset and tasks, we hosted a competition in the KDD CUP 2023 and have attracted thousands of users and submissions. The winning solutions and the associated workshop can be accessed at our website https://kddcup23.github.io/.

{{</citation>}}


## cs.CL (17)



### (56/106) FinGPT: Democratizing Internet-scale Data for Financial Large Language Models (Xiao-Yang Liu et al., 2023)

{{<citation>}}

Xiao-Yang Liu, Guoxuan Wang, Daochen Zha. (2023)  
**FinGPT: Democratizing Internet-scale Data for Financial Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, q-fin-GN  
Keywords: AI, Financial, GPT, Language Model, NLP, Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2307.10485v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable proficiency in understanding and generating human-like texts, which may potentially revolutionize the finance industry. However, existing LLMs often fall short in the financial field, which is mainly attributed to the disparities between general text data and financial text data. Unfortunately, there is only a limited number of financial text datasets available (quite small size), and BloombergGPT, the first financial LLM (FinLLM), is close-sourced (only the training logs were released). In light of this, we aim to democratize Internet-scale financial data for LLMs, which is an open challenge due to diverse data sources, low signal-to-noise ratio, and high time-validity. To address the challenges, we introduce an open-sourced and data-centric framework, \textit{Financial Generative Pre-trained Transformer (FinGPT)}, that automates the collection and curation of real-time financial data from >34 diverse sources on the Internet, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. Additionally, we propose a simple yet effective strategy for fine-tuning FinLLM using the inherent feedback from the market, dubbed Reinforcement Learning with Stock Prices (RLSP). We also adopt the Low-rank Adaptation (LoRA, QLoRA) method that enables users to customize their own FinLLMs from open-source general-purpose LLMs at a low cost. Finally, we showcase several FinGPT applications, including robo-advisor, sentiment analysis for algorithmic trading, and low-code development. FinGPT aims to democratize FinLLMs, stimulate innovation, and unlock new opportunities in open finance. The codes are available at https://github.com/AI4Finance-Foundation/FinGPT and https://github.com/AI4Finance-Foundation/FinNLP

{{</citation>}}


### (57/106) Findings of Factify 2: Multimodal Fake News Detection (S Suryavardan et al., 2023)

{{<citation>}}

S Suryavardan, Shreyash Mishra, Megha Chakraborty, Parth Patwa, Anku Rani, Aman Chadha, Aishwarya Reganti, Amitava Das, Amit Sheth, Manoj Chinnakotla, Asif Ekbal, Srijan Kumar. (2023)  
**Findings of Factify 2: Multimodal Fake News Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: AI, BERT, Fake News  
[Paper Link](http://arxiv.org/abs/2307.10475v1)  

---


**ABSTRACT**  
With social media usage growing exponentially in the past few years, fake news has also become extremely prevalent. The detrimental impact of fake news emphasizes the need for research focused on automating the detection of false information and verifying its accuracy. In this work, we present the outcome of the Factify 2 shared task, which provides a multi-modal fact verification and satire news dataset, as part of the DeFactify 2 workshop at AAAI'23. The data calls for a comparison based approach to the task by pairing social media claims with supporting documents, with both text and image, divided into 5 classes based on multi-modal relations. In the second iteration of this task we had over 60 participants and 9 final test-set submissions. The best performances came from the use of DeBERTa for text and Swinv2 and CLIP for image. The highest F1 score averaged for all five classes was 81.82%.

{{</citation>}}


### (58/106) Can Instruction Fine-Tuned Language Models Identify Social Bias through Prompting? (Omkar Dige et al., 2023)

{{<citation>}}

Omkar Dige, Jacob-Junqi Tian, David Emerson, Faiza Khan Khattak. (2023)  
**Can Instruction Fine-Tuned Language Models Identify Social Bias through Prompting?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Bias, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2307.10472v1)  

---


**ABSTRACT**  
As the breadth and depth of language model applications continue to expand rapidly, it is increasingly important to build efficient frameworks for measuring and mitigating the learned or inherited social biases of these models. In this paper, we present our work on evaluating instruction fine-tuned language models' ability to identify bias through zero-shot prompting, including Chain-of-Thought (CoT) prompts. Across LLaMA and its two instruction fine-tuned versions, Alpaca 7B performs best on the bias identification task with an accuracy of 56.7%. We also demonstrate that scaling up LLM size and data diversity could lead to further performance gain. This is a work-in-progress presenting the first component of our bias mitigation framework. We will keep updating this work as we get more results.

{{</citation>}}


### (59/106) Improving Pre-trained Language Models' Generalization (Somayeh Ghanbarzadeh et al., 2023)

{{<citation>}}

Somayeh Ghanbarzadeh, Hamid Palangi, Yan Huang, Radames Cruz Moreno, Hamed Khanpour. (2023)  
**Improving Pre-trained Language Models' Generalization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.10457v1)  

---


**ABSTRACT**  
The reusability of state-of-the-art Pre-trained Language Models (PLMs) is often limited by their generalization problem, where their performance drastically decreases when evaluated on examples that differ from the training dataset, known as Out-of-Distribution (OOD)/unseen examples. This limitation arises from PLMs' reliance on spurious correlations, which work well for frequent example types but not for general examples. To address this issue, we propose a training approach called Mask-tuning, which integrates Masked Language Modeling (MLM) training objectives into the fine-tuning process to enhance PLMs' generalization. Comprehensive experiments demonstrate that Mask-tuning surpasses current state-of-the-art techniques and enhances PLMs' generalization on OOD datasets while improving their performance on in-distribution datasets. The findings suggest that Mask-tuning improves the reusability of PLMs on unseen data, making them more practical and effective for real-world applications.

{{</citation>}}


### (60/106) Thrust: Adaptively Propels Large Language Models with External Knowledge (Xinran Zhao et al., 2023)

{{<citation>}}

Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, Jianshu Chen. (2023)  
**Thrust: Adaptively Propels Large Language Models with External Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.10442v1)  

---


**ABSTRACT**  
Although large-scale pre-trained language models (PTLMs) are shown to encode rich knowledge in their model parameters, the inherent knowledge in PTLMs can be opaque or static, making external knowledge necessary. However, the existing information retrieval techniques could be costly and may even introduce noisy and sometimes misleading knowledge. To address these challenges, we propose the instance-level adaptive propulsion of external knowledge (IAPEK), where we only conduct the retrieval when necessary. To achieve this goal, we propose measuring whether a PTLM contains enough knowledge to solve an instance with a novel metric, Thrust, which leverages the representation distribution of a small number of seen instances. Extensive experiments demonstrate that thrust is a good measurement of PTLM models' instance-level knowledgeability. Moreover, we can achieve significantly higher cost-efficiency with the Thrust score as the retrieval indicator than the naive usage of external knowledge on 88% of the evaluated tasks with 26% average performance improvement. Such findings shed light on the real-world practice of knowledge-enhanced LMs with a limited knowledge-seeking budget due to computation latency or costs.

{{</citation>}}


### (61/106) PharmacyGPT: The AI Pharmacist (Zhengliang Liu et al., 2023)

{{<citation>}}

Zhengliang Liu, Zihao Wu, Mengxuan Hu, Bokai Zhao, Lin Zhao, Tianyi Zhang, Haixing Dai, Xianyan Chen, Ye Shen, Sheng Li, Brian Murray, Tianming Liu, Andrea Sikora. (2023)  
**PharmacyGPT: The AI Pharmacist**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.10432v2)  

---


**ABSTRACT**  
In this study, we introduce PharmacyGPT, a novel framework to assess the capabilities of large language models (LLMs) such as ChatGPT and GPT-4 in emulating the role of clinical pharmacists. Our methodology encompasses the utilization of LLMs to generate comprehensible patient clusters, formulate medication plans, and forecast patient outcomes. We conduct our investigation using real data acquired from the intensive care unit (ICU) at the University of North Carolina Chapel Hill (UNC) Hospital. Our analysis offers valuable insights into the potential applications and limitations of LLMs in the field of clinical pharmacy, with implications for both patient care and the development of future AI-driven healthcare solutions. By evaluating the performance of PharmacyGPT, we aim to contribute to the ongoing discourse surrounding the integration of artificial intelligence in healthcare settings, ultimately promoting the responsible and efficacious use of such technologies.

{{</citation>}}


### (62/106) DialogStudio: Towards Richest and Most Diverse Unified Dataset Collection for Conversational AI (Jianguo Zhang et al., 2023)

{{<citation>}}

Jianguo Zhang, Kun Qian, Zhiwei Liu, Shelby Heinecke, Rui Meng, Ye Liu, Zhou Yu, Huan Wang, Silvio Savarese, Caiming Xiong. (2023)  
**DialogStudio: Towards Richest and Most Diverse Unified Dataset Collection for Conversational AI**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Dialog  
[Paper Link](http://arxiv.org/abs/2307.10172v2)  

---


**ABSTRACT**  
Despite advancements in conversational AI, language models encounter challenges to handle diverse conversational tasks, and existing dialogue dataset collections often lack diversity and comprehensiveness. To tackle these issues, we introduce DialogStudio: the largest and most diverse collection of dialogue datasets, unified under a consistent format while preserving their original information. Our collection encompasses data from open-domain dialogues, task-oriented dialogues, natural language understanding, conversational recommendation, dialogue summarization, and knowledge-grounded dialogues, making it an incredibly rich and diverse resource for dialogue research and model training. To further enhance the utility of DialogStudio, we identify the licenses for each dataset and design domain-aware prompts for selected dialogues to facilitate instruction-aware fine-tuning. Furthermore, we develop conversational AI models using the dataset collection, and our experiments in both zero-shot and few-shot learning scenarios demonstrate the superiority of DialogStudio. To improve transparency and support dataset and task-based research, as well as language model pre-training, all datasets, licenses, codes, and models associated with DialogStudio are made publicly accessible at https://github.com/salesforce/DialogStudio

{{</citation>}}


### (63/106) Challenges and Applications of Large Language Models (Jean Kaddour et al., 2023)

{{<citation>}}

Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, Robert McHardy. (2023)  
**Challenges and Applications of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.10169v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) went from non-existent to ubiquitous in the machine learning discourse within a few years. Due to the fast pace of the field, it is difficult to identify the remaining challenges and already fruitful application areas. In this paper, we aim to establish a systematic set of open problems and application successes so that ML researchers can comprehend the field's current state more quickly and become productive.

{{</citation>}}


### (64/106) Exploring Transformer Extrapolation (Zhen Qin et al., 2023)

{{<citation>}}

Zhen Qin, Yiran Zhong, Hui Deng. (2023)  
**Exploring Transformer Extrapolation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2307.10156v1)  

---


**ABSTRACT**  
Length extrapolation has attracted considerable attention recently since it allows transformers to be tested on longer sequences than those used in training. Previous research has shown that this property can be attained by using carefully designed Relative Positional Encodings (RPEs). While these methods perform well on a variety of corpora, the conditions for length extrapolation have yet to be investigated. This paper attempts to determine what types of RPEs allow for length extrapolation through a thorough mathematical and empirical analysis. We discover that a transformer is certain to possess this property as long as the series that corresponds to the RPE's exponential converges. Two practices are derived from the conditions and examined in language modeling tasks on a variety of corpora. As a bonus from the conditions, we derive a new Theoretical Receptive Field (TRF) to measure the receptive field of RPEs without taking any training steps. Extensive experiments are conducted on the Wikitext-103, Books, Github, and WikiBook datasets to demonstrate the viability of our discovered conditions. We also compare TRF to Empirical Receptive Field (ERF) across different models, showing consistently matched trends on the aforementioned datasets. The code is available at https://github.com/OpenNLPLab/Rpe.

{{</citation>}}


### (65/106) Gradient Sparsification For Masked Fine-Tuning of Transformers (James O' Neill et al., 2023)

{{<citation>}}

James O' Neill, Sourav Dutta. (2023)  
**Gradient Sparsification For Masked Fine-Tuning of Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GLUE, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.10098v1)  

---


**ABSTRACT**  
Fine-tuning pretrained self-supervised language models is widely adopted for transfer learning to downstream tasks. Fine-tuning can be achieved by freezing gradients of the pretrained network and only updating gradients of a newly added classification layer, or by performing gradient updates on all parameters. Gradual unfreezing makes a trade-off between the two by gradually unfreezing gradients of whole layers during training. This has been an effective strategy to trade-off between storage and training speed with generalization performance. However, it is not clear whether gradually unfreezing layers throughout training is optimal, compared to sparse variants of gradual unfreezing which may improve fine-tuning performance. In this paper, we propose to stochastically mask gradients to regularize pretrained language models for improving overall fine-tuned performance. We introduce GradDrop and variants thereof, a class of gradient sparsification methods that mask gradients during the backward pass, acting as gradient noise. GradDrop is sparse and stochastic unlike gradual freezing. Extensive experiments on the multilingual XGLUE benchmark with XLMR-Large show that GradDrop is competitive against methods that use additional translated data for intermediate pretraining and outperforms standard fine-tuning and gradual unfreezing. A post-analysis shows how GradDrop improves performance with languages it was not trained on, such as under-resourced languages.

{{</citation>}}


### (66/106) Generating Mathematical Derivations with Large Language Models (Jordan Meadows et al., 2023)

{{<citation>}}

Jordan Meadows, Marco Valentino, Andre Freitas. (2023)  
**Generating Mathematical Derivations with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, math-HO  
Keywords: GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2307.09998v1)  

---


**ABSTRACT**  
The derivation of mathematical results in specialised fields using Large Language Models (LLMs) is an emerging research direction that can help identify models' limitations, and potentially support mathematical discovery. In this paper, we leverage a symbolic engine to generate derivations of equations at scale, and investigate the capabilities of LLMs when deriving goal equations from premises. Specifically, we employ in-context learning for GPT and fine-tune a range of T5 models to compare the robustness and generalisation of pre-training strategies to specialised models. Empirical results show that fine-tuned FLAN-T5-large (MathT5) outperforms GPT models on all static and out-of-distribution test sets in terms of absolute performance. However, an in-depth analysis reveals that the fine-tuned models are more sensitive to perturbations involving unseen symbols and (to a lesser extent) changes to equation structure. In addition, we analyse 1.7K equations and over 200 derivations to highlight common reasoning errors such as the inclusion of incorrect, irrelevant, and redundant equations, along with the tendency to skip derivation steps. Finally, we explore the suitability of existing metrics for evaluating mathematical derivations finding evidence that, while they capture general properties such as sensitivity to perturbations, they fail to highlight fine-grained reasoning errors and essential differences between models. Overall, this work demonstrates that training models on synthetic data can improve their mathematical capabilities beyond larger architectures.

{{</citation>}}


### (67/106) GUIDO: A Hybrid Approach to Guideline Discovery & Ordering from Natural Language Texts (Nils Freyer et al., 2023)

{{<citation>}}

Nils Freyer, Dustin Thewes, Matthias Meinecke. (2023)  
**GUIDO: A Hybrid Approach to Guideline Discovery & Ordering from Natural Language Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.09959v1)  

---


**ABSTRACT**  
Extracting workflow nets from textual descriptions can be used to simplify guidelines or formalize textual descriptions of formal processes like business processes and algorithms. The task of manually extracting processes, however, requires domain expertise and effort. While automatic process model extraction is desirable, annotating texts with formalized process models is expensive. Therefore, there are only a few machine-learning-based extraction approaches. Rule-based approaches, in turn, require domain specificity to work well and can rarely distinguish relevant and irrelevant information in textual descriptions. In this paper, we present GUIDO, a hybrid approach to the process model extraction task that first, classifies sentences regarding their relevance to the process model, using a BERT-based sentence classifier, and second, extracts a process model from the sentences classified as relevant, using dependency parsing. The presented approach achieves significantly better results than a pure rule-based approach. GUIDO achieves an average behavioral similarity score of $0.93$. Still, in comparison to purely machine-learning-based approaches, the annotation costs stay low.

{{</citation>}}


### (68/106) Large Language Models can accomplish Business Process Management Tasks (Michael Grohs et al., 2023)

{{<citation>}}

Michael Grohs, Luka Abb, Nourhan Elsayed, Jana-Rebecca Rehse. (2023)  
**Large Language Models can accomplish Business Process Management Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.09923v1)  

---


**ABSTRACT**  
Business Process Management (BPM) aims to improve organizational activities and their outcomes by managing the underlying processes. To achieve this, it is often necessary to consider information from various sources, including unstructured textual documents. Therefore, researchers have developed several BPM-specific solutions that extract information from textual documents using Natural Language Processing techniques. These solutions are specific to their respective tasks and cannot accomplish multiple process-related problems as a general-purpose instrument. However, in light of the recent emergence of Large Language Models (LLMs) with remarkable reasoning capabilities, such a general-purpose instrument with multiple applications now appears attainable. In this paper, we illustrate how LLMs can accomplish text-related BPM tasks by applying a specific LLM to three exemplary tasks: mining imperative process models from textual descriptions, mining declarative process models from textual descriptions, and assessing the suitability of process tasks from textual descriptions for robotic process automation. We show that, without extensive configuration or prompt engineering, LLMs perform comparably to or better than existing solutions and discuss implications for future BPM research as well as practical usage.

{{</citation>}}


### (69/106) Enhancing conversational quality in language learning chatbots: An evaluation of GPT4 for ASR error correction (Long Mai et al., 2023)

{{<citation>}}

Long Mai, Julie Carson-Berndsen. (2023)  
**Enhancing conversational quality in language learning chatbots: An evaluation of GPT4 for ASR error correction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, NLP  
[Paper Link](http://arxiv.org/abs/2307.09744v1)  

---


**ABSTRACT**  
The integration of natural language processing (NLP) technologies into educational applications has shown promising results, particularly in the language learning domain. Recently, many spoken open-domain chatbots have been used as speaking partners, helping language learners improve their language skills. However, one of the significant challenges is the high word-error-rate (WER) when recognizing non-native/non-fluent speech, which interrupts conversation flow and leads to disappointment for learners. This paper explores the use of GPT4 for ASR error correction in conversational settings. In addition to WER, we propose to use semantic textual similarity (STS) and next response sensibility (NRS) metrics to evaluate the impact of error correction models on the quality of the conversation. We find that transcriptions corrected by GPT4 lead to higher conversation quality, despite an increase in WER. GPT4 also outperforms standard error correction methods without the need for in-domain training data.

{{</citation>}}


### (70/106) CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility (Guohai Xu et al., 2023)

{{<citation>}}

Guohai Xu, Jiayi Liu, Ming Yan, Haotian Xu, Jinghui Si, Zhuoran Zhou, Peng Yi, Xing Gao, Jitao Sang, Rong Zhang, Ji Zhang, Chao Peng, Fei Huang, Jingren Zhou. (2023)  
**CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.09705v1)  

---


**ABSTRACT**  
With the rapid evolution of large language models (LLMs), there is a growing concern that they may pose risks or have negative social impacts. Therefore, evaluation of human values alignment is becoming increasingly important. Previous work mainly focuses on assessing the performance of LLMs on certain knowledge and reasoning abilities, while neglecting the alignment to human values, especially in a Chinese context. In this paper, we present CValues, the first Chinese human values evaluation benchmark to measure the alignment ability of LLMs in terms of both safety and responsibility criteria. As a result, we have manually collected adversarial safety prompts across 10 scenarios and induced responsibility prompts from 8 domains by professional experts. To provide a comprehensive values evaluation of Chinese LLMs, we not only conduct human evaluation for reliable comparison, but also construct multi-choice prompts for automatic evaluation. Our findings suggest that while most Chinese LLMs perform well in terms of safety, there is considerable room for improvement in terms of responsibility. Moreover, both the automatic and human evaluation are important for assessing the human values alignment in different aspects. The benchmark and code is available on ModelScope and Github.

{{</citation>}}


### (71/106) Efficient Guided Generation for Large Language Models (Brandon T. Willard et al., 2023)

{{<citation>}}

Brandon T. Willard, Rémi Louf. (2023)  
**Efficient Guided Generation for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.09702v2)  

---


**ABSTRACT**  
In this article we describe an efficient approach to guiding language model text generation with regular expressions and context-free grammars. Our approach adds little to no overhead to the token sequence generation process, and makes guided generation feasible in practice. An implementation is provided in the open source Python library Outlines.

{{</citation>}}


### (72/106) Efficiency Pentathlon: A Standardized Arena for Efficiency Evaluation (Hao Peng et al., 2023)

{{<citation>}}

Hao Peng, Qingqing Cao, Jesse Dodge, Matthew E. Peters, Jared Fernandez, Tom Sherborne, Kyle Lo, Sam Skjonsberg, Emma Strubell, Darrell Plessas, Iz Beltagy, Evan Pete Walsh, Noah A. Smith, Hannaneh Hajishirzi. (2023)  
**Efficiency Pentathlon: A Standardized Arena for Efficiency Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.09701v1)  

---


**ABSTRACT**  
Rising computational demands of modern natural language processing (NLP) systems have increased the barrier to entry for cutting-edge research while posing serious environmental concerns. Yet, progress on model efficiency has been impeded by practical challenges in model evaluation and comparison. For example, hardware is challenging to control due to disparate levels of accessibility across different institutions. Moreover, improvements in metrics such as FLOPs often fail to translate to progress in real-world applications. In response, we introduce Pentathlon, a benchmark for holistic and realistic evaluation of model efficiency. Pentathlon focuses on inference, which accounts for a majority of the compute in a model's lifecycle. It offers a strictly-controlled hardware platform, and is designed to mirror real-world applications scenarios. It incorporates a suite of metrics that target different aspects of efficiency, including latency, throughput, memory overhead, and energy consumption. Pentathlon also comes with a software library that can be seamlessly integrated into any codebase and enable evaluation. As a standardized and centralized evaluation platform, Pentathlon can drastically reduce the workload to make fair and reproducible efficiency comparisons. While initially focused on natural language processing (NLP) models, Pentathlon is designed to allow flexible extension to other fields. We envision Pentathlon will stimulate algorithmic innovations in building efficient models, and foster an increased awareness of the social and environmental implications in the development of future-generation NLP models.

{{</citation>}}


## cs.HC (1)



### (73/106) Towards Sustainable Research Data Management in Human-Computer Interaction (David Goedicke et al., 2023)

{{<citation>}}

David Goedicke, Mark Colley, Sebastian S. Feger, Michael Goedicke, Bastian Pfleging, Wendy Ju. (2023)  
**Towards Sustainable Research Data Management in Human-Computer Interaction**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10467v1)  

---


**ABSTRACT**  
We discuss important aspects of HCI research regarding Research Data Management (RDM) to achieve better publication processes and higher reuse of HCI research results. Various context elements of RDM for HCI are discussed, including examples of existing and emerging infrastructures for RDM. We briefly discuss existing approaches and come up with additional aspects which need to be addressed. This is to apply the so-called FAIR principle fully, which -- besides being findable and accessible -- also includes interoperability and reusability. We also discuss briefly the kind of research data types that play a role here and propose to build on existing work and involve the HCI scientific community to improve current practices.

{{</citation>}}


## cs.AI (7)



### (74/106) A data science axiology: the nature, value, and risks of data science (Michael L. Brodie, 2023)

{{<citation>}}

Michael L. Brodie. (2023)  
**A data science axiology: the nature, value, and risks of data science**  

---
Primary Category: cs.AI  
Categories: I-2; I-2-4; I-2-7; K-2, cs-AI, cs-DB, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10460v1)  

---


**ABSTRACT**  
Data science is not a science. It is a research paradigm with an unfathomed scope, scale, complexity, and power for knowledge discovery that is not otherwise possible and can be beyond human reasoning. It is changing our world practically and profoundly already widely deployed in tens of thousands of applications in every discipline in an AI Arms Race that, due to its inscrutability, can lead to unfathomed risks. This paper presents an axiology of data science, its purpose, nature, importance, risks, and value for problem solving, by exploring and evaluating its remarkable, definitive features. As data science is in its infancy, this initial, speculative axiology is intended to aid in understanding and defining data science to recognize its potential benefits, risks, and open research challenges. AI based data science is inherently about uncertainty that may be more realistic than our preference for the certainty of science. Data science will have impacts far beyond knowledge discovery and will take us into new ways of understanding the world.

{{</citation>}}


### (75/106) Complying with the EU AI Act (Jacintha Walters et al., 2023)

{{<citation>}}

Jacintha Walters, Diptish Dey, Debarati Bhaumik, Sophie Horsman. (2023)  
**Complying with the EU AI Act**  

---
Primary Category: cs.AI  
Categories: I-2, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10458v1)  

---


**ABSTRACT**  
The EU AI Act is the proposed EU legislation concerning AI systems. This paper identifies several categories of the AI Act. Based on this categorization, a questionnaire is developed that serves as a tool to offer insights by creating quantitative data. Analysis of the data shows various challenges for organizations in different compliance categories. The influence of organization characteristics, such as size and sector, is examined to determine the impact on compliance. The paper will also share qualitative data on which questions were prevalent among respondents, both on the content of the AI Act as the application. The paper concludes by stating that there is still room for improvement in terms of compliance with the AIA and refers to a related project that examines a solution to help these organizations.

{{</citation>}}


### (76/106) Chit-Chat or Deep Talk: Prompt Engineering for Process Mining (Urszula Jessen et al., 2023)

{{<citation>}}

Urszula Jessen, Michal Sroka, Dirk Fahland. (2023)  
**Chit-Chat or Deep Talk: Prompt Engineering for Process Mining**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.09909v1)  

---


**ABSTRACT**  
This research investigates the application of Large Language Models (LLMs) to augment conversational agents in process mining, aiming to tackle its inherent complexity and diverse skill requirements. While LLM advancements present novel opportunities for conversational process mining, generating efficient outputs is still a hurdle. We propose an innovative approach that amend many issues in existing solutions, informed by prior research on Natural Language Processing (NLP) for conversational agents. Leveraging LLMs, our framework improves both accessibility and agent performance, as demonstrated by experiments on public question and data sets. Our research sets the stage for future explorations into LLMs' role in process mining and concludes with propositions for enhancing LLM memory, implementing real-time user testing, and examining diverse data sets.

{{</citation>}}


### (77/106) PyTAG: Challenges and Opportunities for Reinforcement Learning in Tabletop Games (Martin Balla et al., 2023)

{{<citation>}}

Martin Balla, George E. M. Long, Dominik Jeurissen, James Goodman, Raluca D. Gaina, Diego Perez-Liebana. (2023)  
**PyTAG: Challenges and Opportunities for Reinforcement Learning in Tabletop Games**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09905v1)  

---


**ABSTRACT**  
In recent years, Game AI research has made important breakthroughs using Reinforcement Learning (RL). Despite this, RL for modern tabletop games has gained little to no attention, even when they offer a range of unique challenges compared to video games. To bridge this gap, we introduce PyTAG, a Python API for interacting with the Tabletop Games framework (TAG). TAG contains a growing set of more than 20 modern tabletop games, with a common API for AI agents. We present techniques for training RL agents in these games and introduce baseline results after training Proximal Policy Optimisation algorithms on a subset of games. Finally, we discuss the unique challenges complex modern tabletop games provide, now open to RL research through PyTAG.

{{</citation>}}


### (78/106) Amortised Design Optimization for Item Response Theory (Antti Keurulainen et al., 2023)

{{<citation>}}

Antti Keurulainen, Isak Westerlund, Oskar Keurulainen, Andrew Howes. (2023)  
**Amortised Design Optimization for Item Response Theory**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09891v1)  

---


**ABSTRACT**  
Item Response Theory (IRT) is a well known method for assessing responses from humans in education and psychology. In education, IRT is used to infer student abilities and characteristics of test items from student responses. Interactions with students are expensive, calling for methods that efficiently gather information for inferring student abilities. Methods based on Optimal Experimental Design (OED) are computationally costly, making them inapplicable for interactive applications. In response, we propose incorporating amortised experimental design into IRT. Here, the computational cost is shifted to a precomputing phase by training a Deep Reinforcement Learning (DRL) agent with synthetic data. The agent is trained to select optimally informative test items for the distribution of students, and to conduct amortised inference conditioned on the experiment outcomes. During deployment the agent estimates parameters from data, and suggests the next test item for the student, in close to real-time, by taking into account the history of experiments and outcomes.

{{</citation>}}


### (79/106) A Fast and Map-Free Model for Trajectory Prediction in Traffics (Junhong Xiang et al., 2023)

{{<citation>}}

Junhong Xiang, Jingmin Zhang, Zhixiong Nan. (2023)  
**A Fast and Map-Free Model for Trajectory Prediction in Traffics**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.09831v1)  

---


**ABSTRACT**  
To handle the two shortcomings of existing methods, (i)nearly all models rely on high-definition (HD) maps, yet the map information is not always available in real traffic scenes and HD map-building is expensive and time-consuming and (ii) existing models usually focus on improving prediction accuracy at the expense of reducing computing efficiency, yet the efficiency is crucial for various real applications, this paper proposes an efficient trajectory prediction model that is not dependent on traffic maps. The core idea of our model is encoding single-agent's spatial-temporal information in the first stage and exploring multi-agents' spatial-temporal interactions in the second stage. By comprehensively utilizing attention mechanism, LSTM, graph convolution network and temporal transformer in the two stages, our model is able to learn rich dynamic and interaction information of all agents. Our model achieves the highest performance when comparing with existing map-free methods and also exceeds most map-based state-of-the-art methods on the Argoverse dataset. In addition, our model also exhibits a faster inference speed than the baseline methods.

{{</citation>}}


### (80/106) Absolutist AI (Mitchell Barrington, 2023)

{{<citation>}}

Mitchell Barrington. (2023)  
**Absolutist AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10315v1)  

---


**ABSTRACT**  
This paper argues that training AI systems with absolute constraints -- which forbid certain acts irrespective of the amount of value they might produce -- may make considerable progress on many AI safety problems in principle. First, it provides a guardrail for avoiding the very worst outcomes of misalignment. Second, it could prevent AIs from causing catastrophes for the sake of very valuable consequences, such as replacing humans with a much larger number of beings living at a higher welfare level. Third, it makes systems more corrigible, allowing creators to make corrective interventions in them, such as altering their objective functions or shutting them down. And fourth, it helps systems explore their environment more safely by prohibiting them from exploring especially dangerous acts. I offer a decision-theoretic formalization of an absolute constraints, improving on existing models in the literature, and use this model to prove some results about the training and behavior of absolutist AIs. I conclude by showing that, although absolutist AIs will not maximize expected value, they will not be susceptible to behave irrationally, and they will not (contra coherence arguments) face environmental pressure to become expected-value maximizers.

{{</citation>}}


## stat.ML (1)



### (81/106) A Matrix Ensemble Kalman Filter-based Multi-arm Neural Network to Adequately Approximate Deep Neural Networks (Ved Piyush et al., 2023)

{{<citation>}}

Ved Piyush, Yuchen Yan, Yuzhen Zhou, Yanbin Yin, Souparno Ghosh. (2023)  
**A Matrix Ensemble Kalman Filter-based Multi-arm Neural Network to Adequately Approximate Deep Neural Networks**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-AP, stat-CO, stat-ML, stat.ML  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.10436v1)  

---


**ABSTRACT**  
Deep Learners (DLs) are the state-of-art predictive mechanism with applications in many fields requiring complex high dimensional data processing. Although conventional DLs get trained via gradient descent with back-propagation, Kalman Filter (KF)-based techniques that do not need gradient computation have been developed to approximate DLs. We propose a multi-arm extension of a KF-based DL approximator that can mimic DL when the sample size is too small to train a multi-arm DL. The proposed Matrix Ensemble Kalman Filter-based multi-arm ANN (MEnKF-ANN) also performs explicit model stacking that becomes relevant when the training sample has an unequal-size feature set. Our proposed technique can approximate Long Short-term Memory (LSTM) Networks and attach uncertainty to the predictions obtained from these LSTMs with desirable coverage. We demonstrate how MEnKF-ANN can "adequately" approximate an LSTM network trained to classify what carbohydrate substrates are digested and utilized by a microbiome sample whose genomic sequences consist of polysaccharide utilization loci (PULs) and their encoded genes.

{{</citation>}}


## cs.SE (4)



### (82/106) Technical Challenges of Deploying Reinforcement Learning Agents for Game Testing in AAA Games (Jonas Gillberg et al., 2023)

{{<citation>}}

Jonas Gillberg, Joakim Bergdahl, Alessandro Sestini, Andrew Eakins, Linus Gisslen. (2023)  
**Technical Challenges of Deploying Reinforcement Learning Agents for Game Testing in AAA Games**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.11105v1)  

---


**ABSTRACT**  
Going from research to production, especially for large and complex software systems, is fundamentally a hard problem. In large-scale game production, one of the main reasons is that the development environment can be very different from the final product. In this technical paper we describe an effort to add an experimental reinforcement learning system to an existing automated game testing solution based on scripted bots in order to increase its capacity. We report on how this reinforcement learning system was integrated with the aim to increase test coverage similar to [1] in a set of AAA games including Battlefield 2042 and Dead Space (2023). The aim of this technical paper is to show a use-case of leveraging reinforcement learning in game production and cover some of the largest time sinks anyone who wants to make the same journey for their game may encounter. Furthermore, to help the game industry to adopt this technology faster, we propose a few research directions that we believe will be valuable and necessary for making machine learning, and especially reinforcement learning, an effective tool in game production.

{{</citation>}}


### (83/106) Code Detection for Hardware Acceleration Using Large Language Models (Pablo Antonio Martínez et al., 2023)

{{<citation>}}

Pablo Antonio Martínez, Gregorio Bernabé, José Manuel García. (2023)  
**Code Detection for Hardware Acceleration Using Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-PL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.10348v1)  

---


**ABSTRACT**  
Large language models (LLMs) have been massively applied to many tasks, often surpassing state-of-the-art approaches. While their effectiveness in code generation has been extensively studied (e.g., AlphaCode), their potential for code detection remains unexplored.   This work presents the first analysis of code detection using LLMs. Our study examines essential kernels, including matrix multiplication, convolution, and fast-fourier transform, implemented in C/C++. We propose both a preliminary, naive prompt and a novel prompting strategy for code detection.   Results reveal that conventional prompting achieves great precision but poor accuracy (68.8%, 22.3%, and 79.2% for GEMM, convolution, and FFT, respectively) due to a high number of false positives. Our novel prompting strategy substantially reduces false positives, resulting in excellent overall accuracy (91.1%, 97.9%, and 99.7%, respectively). These results pose a considerable challenge to existing state-of-the-art code detection methods.

{{</citation>}}


### (84/106) Towards green AI-based software systems: an architecture-centric approach (GAISSA) (Silverio Martínez-Fernández et al., 2023)

{{<citation>}}

Silverio Martínez-Fernández, Xavier Franch, Francisco Durán. (2023)  
**Towards green AI-based software systems: an architecture-centric approach (GAISSA)**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09964v1)  

---


**ABSTRACT**  
Nowadays, AI-based systems have achieved outstanding results and have outperformed humans in different domains. However, the processes of training AI models and inferring from them require high computational resources, which pose a significant challenge in the current energy efficiency societal demand. To cope with this challenge, this research project paper describes the main vision, goals, and expected outcomes of the GAISSA project. The GAISSA project aims at providing data scientists and software engineers tool-supported, architecture-centric methods for the modelling and development of green AI-based systems. Although the project is in an initial stage, we describe the current research results, which illustrate the potential to achieve GAISSA objectives.

{{</citation>}}


### (85/106) Are We Ready to Embrace Generative AI for Software Q&A? (Bowen Xu et al., 2023)

{{<citation>}}

Bowen Xu, Thanh-Dat Nguyen, Thanh Le-Cong, Thong Hoang, Jiakun Liu, Kisub Kim, Chen Gong, Changan Niu, Chenyu Wang, Bach Le, David Lo. (2023)  
**Are We Ready to Embrace Generative AI for Software Q&A?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Generative AI, QA  
[Paper Link](http://arxiv.org/abs/2307.09765v1)  

---


**ABSTRACT**  
Stack Overflow, the world's largest software Q&A (SQA) website, is facing a significant traffic drop due to the emergence of generative AI techniques. ChatGPT is banned by Stack Overflow after only 6 days from its release. The main reason provided by the official Stack Overflow is that the answers generated by ChatGPT are of low quality. To verify this, we conduct a comparative evaluation of human-written and ChatGPT-generated answers. Our methodology employs both automatic comparison and a manual study. Our results suggest that human-written and ChatGPT-generated answers are semantically similar, however, human-written answers outperform ChatGPT-generated ones consistently across multiple aspects, specifically by 10% on the overall score. We release the data, analysis scripts, and detailed results at https://anonymous.4open.science/r/GAI4SQA-FD5C.

{{</citation>}}


## cs.RO (3)



### (86/106) Robust Driving Policy Learning with Guided Meta Reinforcement Learning (Kanghoon Lee et al., 2023)

{{<citation>}}

Kanghoon Lee, Jiachen Li, David Isele, Jinkyoo Park, Kikuo Fujimura, Mykel J. Kochenderfer. (2023)  
**Robust Driving Policy Learning with Guided Meta Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.10160v1)  

---


**ABSTRACT**  
Although deep reinforcement learning (DRL) has shown promising results for autonomous navigation in interactive traffic scenarios, existing work typically adopts a fixed behavior policy to control social vehicles in the training environment. This may cause the learned driving policy to overfit the environment, making it difficult to interact well with vehicles with different, unseen behaviors. In this work, we introduce an efficient method to train diverse driving policies for social vehicles as a single meta-policy. By randomizing the interaction-based reward functions of social vehicles, we can generate diverse objectives and efficiently train the meta-policy through guiding policies that achieve specific objectives. We further propose a training strategy to enhance the robustness of the ego vehicle's driving policy using the environment where social vehicles are controlled by the learned meta-policy. Our method successfully learns an ego driving policy that generalizes well to unseen situations with out-of-distribution (OOD) social agents' behaviors in a challenging uncontrolled T-intersection scenario.

{{</citation>}}


### (87/106) BERRY: Bit Error Robustness for Energy-Efficient Reinforcement Learning-Based Autonomous Systems (Zishen Wan et al., 2023)

{{<citation>}}

Zishen Wan, Nandhini Chandramoorthy, Karthik Swaminathan, Pin-Yu Chen, Vijay Janapa Reddi, Arijit Raychowdhury. (2023)  
**BERRY: Bit Error Robustness for Energy-Efficient Reinforcement Learning-Based Autonomous Systems**  

---
Primary Category: cs.RO  
Categories: cs-AR, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.10041v1)  

---


**ABSTRACT**  
Autonomous systems, such as Unmanned Aerial Vehicles (UAVs), are expected to run complex reinforcement learning (RL) models to execute fully autonomous position-navigation-time tasks within stringent onboard weight and power constraints. We observe that reducing onboard operating voltage can benefit the energy efficiency of both the computation and flight mission, however, it can also result in on-chip bit failures that are detrimental to mission safety and performance. To this end, we propose BERRY, a robust learning framework to improve bit error robustness and energy efficiency for RL-enabled autonomous systems. BERRY supports robust learning, both offline and on-board the UAV, and for the first time, demonstrates the practicality of robust low-voltage operation on UAVs that leads to high energy savings in both compute-level operation and system-level quality-of-flight. We perform extensive experiments on 72 autonomous navigation scenarios and demonstrate that BERRY generalizes well across environments, UAVs, autonomy policies, operating voltages and fault patterns, and consistently improves robustness, efficiency and mission performance, achieving up to 15.62% reduction in flight energy, 18.51% increase in the number of successful missions, and 3.43x processing energy reduction.

{{</citation>}}


### (88/106) RobôCIn Small Size League Extended Team Description Paper for RoboCup 2023 (Aline Lima de Oliveira et al., 2023)

{{<citation>}}

Aline Lima de Oliveira, Cauê Addae da Silva Gomes, Cecília Virginia Santos da Silva, Charles Matheus de Sousa Alves, Danilo Andrade Martins de Souza, Driele Pires Ferreira Araújo Xavier, Edgleyson Pereira da Silva, Felipe Bezerra Martins, Lucas Henrique Cavalcanti Santos, Lucas Dias Maciel, Matheus Paixão Gumercindo dos Santos, Matheus Lafayette Vasconcelos, Matheus Vinícius Teotonio do Nascimento Andrade, João Guilherme Oliveira Carvalho de Melo, João Pedro Souza Pereira de Moura, José Ronald da Silva, José Victor Silva Cruz, Pedro Henrique Santana de Morais, Pedro Paulo Salman de Oliveira, Riei Joaquim Matos Rodrigues, Roberto Costa Fernandes, Ryan Vinicius Santos Morais, Tamara Mayara Ramos Teobaldo, Washington Igor dos Santos Silva, Edna Natividade Silva Barros. (2023)  
**RobôCIn Small Size League Extended Team Description Paper for RoboCup 2023**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10018v1)  

---


**ABSTRACT**  
Rob\^oCIn has participated in RoboCup Small Size League since 2019, won its first world title in 2022 (Division B), and is currently a three-times Latin-American champion. This paper presents our improvements to defend the Small Size League (SSL) division B title in RoboCup 2023 in Bordeaux, France. This paper aims to share some of the academic research that our team developed over the past year. Our team has successfully published 2 articles related to SSL at two high-impact conferences: the 25th RoboCup International Symposium and the 19th IEEE Latin American Robotics Symposium (LARS 2022). Over the last year, we have been continuously migrating from our past codebase to Unification. We will describe the new architecture implemented and some points of software and AI refactoring. In addition, we discuss the process of integrating machined components into the mechanical system, our development for participating in the vision blackout challenge last year and what we are preparing for this year.

{{</citation>}}


## cs.SI (3)



### (89/106) Twits, Toxic Tweets, and Tribal Tendencies: Trends in Politically Polarized Posts on Twitter (Hans W. A. Hanley et al., 2023)

{{<citation>}}

Hans W. A. Hanley, Zakir Durumeric. (2023)  
**Twits, Toxic Tweets, and Tribal Tendencies: Trends in Politically Polarized Posts on Twitter**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: BERT, Google, Twitter  
[Paper Link](http://arxiv.org/abs/2307.10349v1)  

---


**ABSTRACT**  
Social media platforms are often blamed for exacerbating political polarization and worsening public dialogue. Many claim hyperpartisan users post pernicious content, slanted to their political views, inciting contentious and toxic conversations. However, what factors, actually contribute to increased online toxicity and negative interactions? In this work, we explore the role that political ideology plays in contributing to toxicity both on an individual user level and a topic level on Twitter. To do this, we train and open-source a DeBERTa-based toxicity detector with a contrastive objective that outperforms the Google Jigsaw Persective Toxicity detector on the Civil Comments test dataset. Then, after collecting 187 million tweets from 55,415 Twitter users, we determine how several account-level characteristics, including political ideology and account age, predict how often each user posts toxic content. Running a linear regression, we find that the diversity of views and the toxicity of the other accounts with which that user engages has a more marked effect on their own toxicity. Namely, toxic comments are correlated with users who engage with a wider array of political views. Performing topic analysis on the toxic content posted by these accounts using the large language model MPNet and a version of the DP-Means clustering algorithm, we find similar behavior across 6,592 individual topics, with conversations on each topic becoming more toxic as a wider diversity of users become involved.

{{</citation>}}


### (90/106) Are you in a Masquerade? Exploring the Behavior and Impact of Large Language Model Driven Social Bots in Online Social Networks (Siyu Li et al., 2023)

{{<citation>}}

Siyu Li, Jin Yang, Kui Zhao. (2023)  
**Are you in a Masquerade? Exploring the Behavior and Impact of Large Language Model Driven Social Bots in Online Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Language Model, Social Network, Twitter  
[Paper Link](http://arxiv.org/abs/2307.10337v1)  

---


**ABSTRACT**  
As the capabilities of Large Language Models (LLMs) emerge, they not only assist in accomplishing traditional tasks within more efficient paradigms but also stimulate the evolution of social bots. Researchers have begun exploring the implementation of LLMs as the driving core of social bots, enabling more efficient and user-friendly completion of tasks like profile completion, social behavior decision-making, and social content generation. However, there is currently a lack of systematic research on the behavioral characteristics of LLMs-driven social bots and their impact on social networks. We have curated data from Chirper, a Twitter-like social network populated by LLMs-driven social bots and embarked on an exploratory study. Our findings indicate that: (1) LLMs-driven social bots possess enhanced individual-level camouflage while exhibiting certain collective characteristics; (2) these bots have the ability to exert influence on online communities through toxic behaviors; (3) existing detection methods are applicable to the activity environment of LLMs-driven social bots but may be subject to certain limitations in effectiveness. Moreover, we have organized the data collected in our study into the Masquerade-23 dataset, which we have publicly released, thus addressing the data void in the subfield of LLMs-driven social bots behavior datasets. Our research outcomes provide primary insights for the research and governance of LLMs-driven social bots within the research community.

{{</citation>}}


### (91/106) Analyzing large scale political discussions on Twitter: the use case of the Greek wiretapping scandal (#ypoklopes) (Ilias Dimitriadis et al., 2023)

{{<citation>}}

Ilias Dimitriadis, Dimitrios P. Giakatos, Stelios Karamanidis, Pavlos Sermpezis, Kelly Kiki, Athena Vakali. (2023)  
**Analyzing large scale political discussions on Twitter: the use case of the Greek wiretapping scandal (#ypoklopes)**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.09819v1)  

---


**ABSTRACT**  
In this paper, we study the Greek wiretappings scandal, which has been revealed in 2022 and attracted a lot of attention by press and citizens. Specifically, we propose a methodology for collecting data and analyzing patterns of online public discussions on Twitter. We apply our methodology to the Greek wiretappings use case, and present findings related to the evolution of the discussion over time, its polarization, and the role of the media. The methodology can be of wider use and replicated to other topics. Finally, we provide publicly an open dataset, and online resources with the results.

{{</citation>}}


## eess.AS (1)



### (92/106) Alzheimer's Disease Detection from Spontaneous Speech and Text: A review (Vrindha M. K. et al., 2023)

{{<citation>}}

Vrindha M. K., Geethu V., Anurenjan P. R., Deepak S., Sreeni K. G.. (2023)  
**Alzheimer's Disease Detection from Spontaneous Speech and Text: A review**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2307.10005v1)  

---


**ABSTRACT**  
In the past decade, there has been a surge in research examining the use of voice and speech analysis as a means of detecting neurodegenerative diseases such as Alzheimer's. Many studies have shown that certain acoustic features can be used to differentiate between normal aging and Alzheimer's disease, and speech analysis has been found to be a cost-effective method of detecting Alzheimer's dementia. The aim of this review is to analyze the various algorithms used in speech-based detection and classification of Alzheimer's disease. A literature survey was conducted using databases such as Web of Science, Google Scholar, and Science Direct, and articles published from January 2020 to the present were included based on keywords such as ``Alzheimer's detection'', "speech," and "natural language processing." The ADReSS, Pitt corpus, and CCC datasets are commonly used for the analysis of dementia from speech, and this review focuses on the various acoustic and linguistic feature engineering-based classification models drawn from 15 studies.   Based on the findings of this study, it appears that a more accurate model for classifying Alzheimer's disease can be developed by considering both linguistic and acoustic data. The review suggests that speech signals can be a useful tool for detecting dementia and may serve as a reliable biomarker for efficiently identifying Alzheimer's disease.

{{</citation>}}


## eess.SY (1)



### (93/106) Solving scalability issues in calculating PV hosting capacity in low voltage distribution networks (Tomislav Antic et al., 2023)

{{<citation>}}

Tomislav Antic, Andrew Keane, Tomislav Capuder. (2023)  
**Solving scalability issues in calculating PV hosting capacity in low voltage distribution networks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY, math-OC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.09971v1)  

---


**ABSTRACT**  
The share of end-users with installed rooftop photovoltaic (PV) systems is continuously growing. Since most end-users are located at the low voltage (LV) level and due to technical limitations of LV networks, it is necessary to calculate PV hosting capacity. Most approaches in calculating a network's hosting capacity are based on three-phase optimal power flow (OPF) formulations. Linearized and relaxed three-phase OPF formulations respectively lose their accuracy and exactness when applied to solve the hosting capacity problem, and only non-linear programming (NLP) models guarantee the exact solution. Compared to linearized or relaxed models, NLP models require a higher computational time for finding an optimal solution. The binary variables uplift the problem to mixed-integer (MI)NLP and increase the computational burden. To resolve the scalability issues in calculating the hosting capacity of single-phase connected PVs, we propose a method that does not entail binary variables but still ensures that PVs are not connected to more than one phase at a time. Due to a risk of a sub-optimal solution, the proposed approach is compared to the results obtained by the MINLP formulation. The comparison includes values of the solution time and technical quantities such as network losses, voltage deviations, and voltage unbalance factor.

{{</citation>}}


## cs.NI (3)



### (94/106) Bias in Internet Measurement Platforms (Pavlos Sermpezis et al., 2023)

{{<citation>}}

Pavlos Sermpezis, Lars Prehn, Sofia Kostoglou, Marcel Flores, Athena Vakali, Emile Aben. (2023)  
**Bias in Internet Measurement Platforms**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.09958v1)  

---


**ABSTRACT**  
Network operators and researchers frequently use Internet measurement platforms (IMPs), such as RIPE Atlas, RIPE RIS, or RouteViews for, e.g., monitoring network performance, detecting routing events, topology discovery, or route optimization. To interpret the results of their measurements and avoid pitfalls or wrong generalizations, users must understand a platform's limitations. To this end, this paper studies an important limitation of IMPs, the \textit{bias}, which exists due to the non-uniform deployment of the vantage points. Specifically, we introduce a generic framework to systematically and comprehensively quantify the multi-dimensional (e.g., across location, topology, network types, etc.) biases of IMPs. Using the framework and open datasets, we perform a detailed analysis of biases in IMPs that confirms well-known (to the domain experts) biases and sheds light on less-known or unexplored biases. To facilitate IMP users to obtain awareness of and explore bias in their measurements, as well as further research and analyses (e.g., methods for mitigating bias), we publicly share our code and data, and provide online tools (API, Web app, etc.) that calculate and visualize the bias in measurement setups.

{{</citation>}}


### (95/106) A3D: Adaptive, Accurate, and Autonomous Navigation for Edge-Assisted Drones (Liekang Zeng et al., 2023)

{{<citation>}}

Liekang Zeng, Haowei Chen, Daipeng Feng, Xiaoxi Zhang, Xu Chen. (2023)  
**A3D: Adaptive, Accurate, and Autonomous Navigation for Edge-Assisted Drones**  

---
Primary Category: cs.NI  
Categories: cs-CV, cs-DC, cs-NI, cs-RO, cs.NI  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.09880v1)  

---


**ABSTRACT**  
Accurate navigation is of paramount importance to ensure flight safety and efficiency for autonomous drones. Recent research starts to use Deep Neural Networks to enhance drone navigation given their remarkable predictive capability for visual perception. However, existing solutions either run DNN inference tasks on drones in situ, impeded by the limited onboard resource, or offload the computation to external servers which may incur large network latency. Few works consider jointly optimizing the offloading decisions along with image transmission configurations and adapting them on the fly. In this paper, we propose A3D, an edge server assisted drone navigation framework that can dynamically adjust task execution location, input resolution, and image compression ratio in order to achieve low inference latency, high prediction accuracy, and long flight distances. Specifically, we first augment state-of-the-art convolutional neural networks for drone navigation and define a novel metric called Quality of Navigation as our optimization objective which can effectively capture the above goals. We then design a deep reinforcement learning based neural scheduler at the drone side for which an information encoder is devised to reshape the state features and thus improve its learning ability. To further support simultaneous multi-drone serving, we extend the edge server design by developing a network-aware resource allocation algorithm, which allows provisioning containerized resources aligned with drones' demand. We finally implement a proof-of-concept prototype with realistic devices and validate its performance in a real-world campus scene, as well as a simulation environment for thorough evaluation upon AirSim. Extensive experimental results show that A3D can reduce end-to-end latency by 28.06% and extend the flight distance by up to 27.28% compared with non-adaptive solutions.

{{</citation>}}


### (96/106) Joint Service Caching, Communication and Computing Resource Allocation in Collaborative MEC Systems: A DRL-based Two-timescale Approach (Qianqian Liu et al., 2023)

{{<citation>}}

Qianqian Liu, Haixia Zhang, Xin Zhang, Dongfeng Yuan. (2023)  
**Joint Service Caching, Communication and Computing Resource Allocation in Collaborative MEC Systems: A DRL-based Two-timescale Approach**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.09691v1)  

---


**ABSTRACT**  
Meeting the strict Quality of Service (QoS) requirements of terminals has imposed a signiffcant challenge on Multiaccess Edge Computing (MEC) systems, due to the limited multidimensional resources. To address this challenge, we propose a collaborative MEC framework that facilitates resource sharing between the edge servers, and with the aim to maximize the long-term QoS and reduce the cache switching cost through joint optimization of service caching, collaborative offfoading, and computation and communication resource allocation. The dual timescale feature and temporal recurrence relationship between service caching and other resource allocation make solving the problem even more challenging. To solve it, we propose a deep reinforcement learning (DRL)-based dual timescale scheme, called DGL-DDPG, which is composed of a short-term genetic algorithm (GA) and a long short-term memory network-based deep deterministic policy gradient (LSTM-DDPG). In doing so, we reformulate the optimization problem as a Markov decision process (MDP) where the small-timescale resource allocation decisions generated by an improved GA are taken as the states and input into a centralized LSTM-DDPG agent to generate the service caching decision for the large-timescale. Simulation results demonstrate that our proposed algorithm outperforms the baseline algorithms in terms of the average QoS and cache switching cost.

{{</citation>}}


## cs.CY (2)



### (97/106) Test-takers have a say: understanding the implications of the use of AI in language tests (Dawen Zhang et al., 2023)

{{<citation>}}

Dawen Zhang, Thong Hoang, Shidong Pan, Yongquan Hu, Zhenchang Xing, Mark Staples, Xiwei Xu, Qinghua Lu, Aaron Quigley. (2023)  
**Test-takers have a say: understanding the implications of the use of AI in language tests**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs.CY  
Keywords: AI, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.09885v1)  

---


**ABSTRACT**  
Language tests measure a person's ability to use a language in terms of listening, speaking, reading, or writing. Such tests play an integral role in academic, professional, and immigration domains, with entities such as educational institutions, professional accreditation bodies, and governments using them to assess candidate language proficiency. Recent advances in Artificial Intelligence (AI) and the discipline of Natural Language Processing have prompted language test providers to explore AI's potential applicability within language testing, leading to transformative activity patterns surrounding language instruction and learning. However, with concerns over AI's trustworthiness, it is imperative to understand the implications of integrating AI into language testing. This knowledge will enable stakeholders to make well-informed decisions, thus safeguarding community well-being and testing integrity. To understand the concerns and effects of AI usage in language tests, we conducted interviews and surveys with English test-takers. To the best of our knowledge, this is the first empirical study aimed at identifying the implications of AI adoption in language tests from a test-taker perspective. Our study reveals test-taker perceptions and behavioral patterns. Specifically, we identify that AI integration may enhance perceptions of fairness, consistency, and availability. Conversely, it might incite mistrust regarding reliability and interactivity aspects, subsequently influencing the behaviors and well-being of test-takers. These insights provide a better understanding of potential societal implications and assist stakeholders in making informed decisions concerning AI usage in language testing.

{{</citation>}}


### (98/106) Unmaking AI Imagemaking: A Methodological Toolkit for Critical Investigation (Luke Munn et al., 2023)

{{<citation>}}

Luke Munn, Liam Magee, Vanicka Arora. (2023)  
**Unmaking AI Imagemaking: A Methodological Toolkit for Critical Investigation**  

---
Primary Category: cs.CY  
Categories: K-4-1; K-2; J-5, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09753v1)  

---


**ABSTRACT**  
AI image models are rapidly evolving, disrupting aesthetic production in many industries. However, understanding of their underlying archives, their logic of image reproduction, and their persistent biases remains limited. What kind of methods and approaches could open up these black boxes? In this paper, we provide three methodological approaches for investigating AI image models and apply them to Stable Diffusion as a case study. Unmaking the ecosystem analyzes the values, structures, and incentives surrounding the model's production. Unmaking the data analyzes the images and text the model draws upon, with their attendant particularities and biases. Unmaking the output analyzes the model's generative results, revealing its logics through prompting, reflection, and iteration. Each mode of inquiry highlights particular ways in which the image model captures, "understands," and recreates the world. This accessible framework supports the work of critically investigating generative AI image models and paves the way for more socially and politically attuned analyses of their impacts in the world.

{{</citation>}}


## math.NA (1)



### (99/106) AutoAMG($θ$): An Auto-tuned AMG Method Based on Deep Learning for Strong Threshold (Haifeng Zou et al., 2023)

{{<citation>}}

Haifeng Zou, Xiaowen Xu, Chen-Song Zhang, Zeyao Mo. (2023)  
**AutoAMG($θ$): An Auto-tuned AMG Method Based on Deep Learning for Strong Threshold**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.09879v1)  

---


**ABSTRACT**  
Algebraic Multigrid (AMG) is one of the most widely used iterative algorithms for solving large sparse linear equations $Ax=b$. In AMG, the coarse grid is a key component that affects the efficiency of the algorithm, the construction of which relies on the strong threshold parameter $\theta$. This parameter is generally chosen empirically, with a default value in many current AMG solvers of 0.25 for 2D problems and 0.5 for 3D problems. However, for many practical problems, the quality of the coarse grid and the efficiency of the AMG algorithm are sensitive to $\theta$; the default value is rarely optimal, and sometimes is far from it. Therefore, how to choose a better $\theta$ is an important question. In this paper, we propose a deep learning based auto-tuning method, AutoAMG($\theta$) for multiscale sparse linear equations, which are widely used in practical problems. The method uses Graph Neural Networks (GNNs) to extract matrix features, and a Multilayer Perceptron (MLP) to build the mapping between matrix features and the optimal $\theta$, which can adaptively output $\theta$ values for different matrices. Numerical experiments show that AutoAMG($\theta$) can achieve significant speedup compared to the default $\theta$ value.

{{</citation>}}


## eess.SP (2)



### (100/106) Introduction to Drone Detection Radar with Emphasis on Automatic Target Recognition (ATR) technology (Jiangkun Gong et al., 2023)

{{<citation>}}

Jiangkun Gong, Jun Yan, Deyong Kong, Deren Li. (2023)  
**Introduction to Drone Detection Radar with Emphasis on Automatic Target Recognition (ATR) technology**  

---
Primary Category: eess.SP  
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.10326v1)  

---


**ABSTRACT**  
This paper discusses the challenges of detecting and categorizing small drones with radar automatic target recognition (ATR) technology. The authors suggest integrating ATR capabilities into drone detection radar systems to improve performance and manage emerging threats. The study focuses primarily on drones in Group 1 and 2. The paper highlights the need to consider kinetic features and signal signatures, such as micro-Doppler, in ATR techniques to efficiently recognize small drones. The authors also present a comprehensive drone detection radar system design that balances detection and tracking requirements, incorporating parameter adjustment based on scattering region theory. They offer an example of a performance improvement achieved using feedback and situational awareness mechanisms with the integrated ATR capabilities. Furthermore, the paper examines challenges related to one-way attack drones and explores the potential of cognitive radar as a solution. The integration of ATR capabilities transforms a 3D radar system into a 4D radar system, resulting in improved drone detection performance. These advancements are useful in military, civilian, and commercial applications, and ongoing research and development efforts are essential to keep radar systems effective and ready to detect, track, and respond to emerging threats.

{{</citation>}}


### (101/106) Perturbing a Neural Network to Infer Effective Connectivity: Evidence from Synthetic EEG Data (Peizhen Yang et al., 2023)

{{<citation>}}

Peizhen Yang, Xinke Shen, Zongsheng Li, Zixiang Luo, Kexin Lou, Quanying Liu. (2023)  
**Perturbing a Neural Network to Infer Effective Connectivity: Evidence from Synthetic EEG Data**  

---
Primary Category: eess.SP  
Categories: cs-AI, eess-SP, eess.SP, q-bio-NC  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09770v1)  

---


**ABSTRACT**  
Identifying causal relationships among distinct brain areas, known as effective connectivity, holds key insights into the brain's information processing and cognitive functions. Electroencephalogram (EEG) signals exhibit intricate dynamics and inter-areal interactions within the brain. However, methods for characterizing nonlinear causal interactions among multiple brain regions remain relatively underdeveloped. In this study, we proposed a data-driven framework to infer effective connectivity by perturbing the trained neural networks. Specifically, we trained neural networks (i.e., CNN, vanilla RNN, GRU, LSTM, and Transformer) to predict future EEG signals according to historical data and perturbed the networks' input to obtain effective connectivity (EC) between the perturbed EEG channel and the rest of the channels. The EC reflects the causal impact of perturbing one node on others. The performance was tested on the synthetic EEG generated by a biological-plausible Jansen-Rit model. CNN and Transformer obtained the best performance on both 3-channel and 90-channel synthetic EEG data, outperforming the classical Granger causality method. Our work demonstrated the potential of perturbing an artificial neural network, learned to predict future system dynamics, to uncover the underlying causal structure.

{{</citation>}}


## q-fin.TR (1)



### (102/106) Reinforcement Learning for Credit Index Option Hedging (Francesco Mandelli et al., 2023)

{{<citation>}}

Francesco Mandelli, Marco Pinciroli, Michele Trapletti, Edoardo Vittori. (2023)  
**Reinforcement Learning for Credit Index Option Hedging**  

---
Primary Category: q-fin.TR  
Categories: cs-LG, q-fin-CP, q-fin-RM, q-fin-TR, q-fin.TR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09844v1)  

---


**ABSTRACT**  
In this paper, we focus on finding the optimal hedging strategy of a credit index option using reinforcement learning. We take a practical approach, where the focus is on realism i.e. discrete time, transaction costs; even testing our policy on real market data. We apply a state of the art algorithm, the Trust Region Volatility Optimization (TRVO) algorithm and show that the derived hedging strategy outperforms the practitioner's Black & Scholes delta hedge.

{{</citation>}}


## cs.SD (3)



### (103/106) From West to East: Who can understand the music of the others better? (Charilaos Papaioannou et al., 2023)

{{<citation>}}

Charilaos Papaioannou, Emmanouil Benetos, Alexandros Potamianos. (2023)  
**From West to East: Who can understand the music of the others better?**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.09795v1)  

---


**ABSTRACT**  
Recent developments in MIR have led to several benchmark deep learning models whose embeddings can be used for a variety of downstream tasks. At the same time, the vast majority of these models have been trained on Western pop/rock music and related styles. This leads to research questions on whether these models can be used to learn representations for different music cultures and styles, or whether we can build similar music audio embedding models trained on data from different cultures or styles. To that end, we leverage transfer learning methods to derive insights about the similarities between the different music cultures to which the data belongs to. We use two Western music datasets, two traditional/folk datasets coming from eastern Mediterranean cultures, and two datasets belonging to Indian art music. Three deep audio embedding models are trained and transferred across domains, including two CNN-based and a Transformer-based architecture, to perform auto-tagging for each target domain dataset. Experimental results show that competitive performance is achieved in all domains via transfer learning, while the best source dataset varies for each music culture. The implementation and the trained models are both provided in a public repository.

{{</citation>}}


### (104/106) Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls (Lejun Min et al., 2023)

{{<citation>}}

Lejun Min, Junyan Jiang, Gus Xia, Jingwei Zhao. (2023)  
**Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.10304v1)  

---


**ABSTRACT**  
We propose Polyffusion, a diffusion model that generates polyphonic music scores by regarding music as image-like piano roll representations. The model is capable of controllable music generation with two paradigms: internal control and external control. Internal control refers to the process in which users pre-define a part of the music and then let the model infill the rest, similar to the task of masked music generation (or music inpainting). External control conditions the model with external yet related information, such as chord, texture, or other features, via the cross-attention mechanism. We show that by using internal and external controls, Polyffusion unifies a wide range of music creation tasks, including melody generation given accompaniment, accompaniment generation given melody, arbitrary music segment inpainting, and music arrangement given chords or textures. Experimental results show that our model significantly outperforms existing Transformer and sampling-based baselines, and using pre-trained disentangled representations as external conditions yields more effective controls.

{{</citation>}}


### (105/106) Improving Domain Generalization for Sound Classification with Sparse Frequency-Regularized Transformer (Honglin Mu et al., 2023)

{{<citation>}}

Honglin Mu, Wentian Xia, Wanxiang Che. (2023)  
**Improving Domain Generalization for Sound Classification with Sparse Frequency-Regularized Transformer**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09723v1)  

---


**ABSTRACT**  
Sound classification models' performance suffers from generalizing on out-of-distribution (OOD) data. Numerous methods have been proposed to help the model generalize. However, most either introduce inference overheads or focus on long-lasting CNN-variants, while Transformers has been proven to outperform CNNs on numerous natural language processing and computer vision tasks. We propose FRITO, an effective regularization technique on Transformer's self-attention, to improve the model's generalization ability by limiting each sequence position's attention receptive field along the frequency dimension on the spectrogram. Experiments show that our method helps Transformer models achieve SOTA generalization performance on TAU 2020 and Nsynth datasets while saving 20% inference time.

{{</citation>}}


## cs.DL (1)



### (106/106) On the Origin of LLMs: An Evolutionary Tree and Graph for 15,821 Large Language Models (Sarah Gao et al., 2023)

{{<citation>}}

Sarah Gao, Andrew Kean Gao. (2023)  
**On the Origin of LLMs: An Evolutionary Tree and Graph for 15,821 Large Language Models**  

---
Primary Category: cs.DL  
Categories: I-2-1; H-5-0, cs-CL, cs-DL, cs.DL  
Keywords: ChatGPT, GPT, Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2307.09793v1)  

---


**ABSTRACT**  
Since late 2022, Large Language Models (LLMs) have become very prominent with LLMs like ChatGPT and Bard receiving millions of users. Hundreds of new LLMs are announced each week, many of which are deposited to Hugging Face, a repository of machine learning models and datasets. To date, nearly 16,000 Text Generation models have been uploaded to the site. Given the huge influx of LLMs, it is of interest to know which LLM backbones, settings, training methods, and families are popular or trending. However, there is no comprehensive index of LLMs available. We take advantage of the relatively systematic nomenclature of Hugging Face LLMs to perform hierarchical clustering and identify communities amongst LLMs using n-grams and term frequency-inverse document frequency. Our methods successfully identify families of LLMs and accurately cluster LLMs into meaningful subgroups. We present a public web application to navigate and explore Constellation, our atlas of 15,821 LLMs. Constellation rapidly generates a variety of visualizations, namely dendrograms, graphs, word clouds, and scatter plots. Constellation is available at the following link: https://constellation.sites.stanford.edu/.

{{</citation>}}
