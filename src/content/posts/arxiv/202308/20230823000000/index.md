---
draft: false
title: "arXiv @ 2023.08.23"
date: 2023-08-23
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.23"
    identifier: arxiv_20230823
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (2)](#csro-2)
- [cs.SD (2)](#cssd-2)
- [cs.SE (5)](#csse-5)
- [cs.CV (38)](#cscv-38)
- [cs.AI (8)](#csai-8)
- [cs.LG (24)](#cslg-24)
- [eess.IV (2)](#eessiv-2)
- [cs.CR (4)](#cscr-4)
- [cs.NI (2)](#csni-2)
- [cs.IT (1)](#csit-1)
- [cs.NE (1)](#csne-1)
- [cs.GR (1)](#csgr-1)
- [cs.CL (20)](#cscl-20)
- [cs.GT (1)](#csgt-1)
- [cs.IR (4)](#csir-4)
- [cs.HC (1)](#cshc-1)
- [quant-ph (1)](#quant-ph-1)
- [q-bio.BM (1)](#q-biobm-1)
- [cs.LO (1)](#cslo-1)
- [cs.PF (1)](#cspf-1)
- [eess.AS (1)](#eessas-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.SI (1)](#cssi-1)
- [stat.ML (1)](#statml-1)
- [cs.MA (1)](#csma-1)

## cs.RO (2)



### (1/125) Embedded Object Detection and Mapping in Soft Materials Using Optical Tactile Sensing (Jose A. Solano-Castellanos et al., 2023)

{{<citation>}}

Jose A. Solano-Castellanos, Won Kyung Do, Monroe Kennedy III. (2023)  
**Embedded Object Detection and Mapping in Soft Materials Using Optical Tactile Sensing**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.11087v1)  

---


**ABSTRACT**  
In this paper, we present a methodology that uses an optical tactile sensor for efficient tactile exploration of embedded objects within soft materials. The methodology consists of an exploration phase, where a probabilistic estimate of the location of the embedded objects is built using a Bayesian approach. The exploration phase is then followed by a mapping phase which exploits the probabilistic map to reconstruct the underlying topography of the workspace by sampling in more detail regions where there is expected to be embedded objects. To demonstrate the effectiveness of the method, we tested our approach on an experimental setup that consists of a series of quartz beads located underneath a polyethylene foam that prevents direct observation of the configuration and requires the use of tactile exploration to recover the location of the beads. We show the performance of our methodology using ten different configurations of the beads where the proposed approach is able to approximate the underlying configuration. We benchmark our results against a random sampling policy.

{{</citation>}}


### (2/125) Reducing Object Detection Uncertainty from RGB and Thermal Data for UAV Outdoor Surveillance (Juan Sandino et al., 2023)

{{<citation>}}

Juan Sandino, Peter A. Caccetta, Conrad Sanderson, Frederic Maire, Felipe Gonzalez. (2023)  
**Reducing Object Detection Uncertainty from RGB and Thermal Data for UAV Outdoor Surveillance**  

---
Primary Category: cs.RO  
Categories: 68T10, 68T40, 68T45, 93C85, B-8-1; C-3; I-5-4; J-2; J-7, cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.10671v1)  

---


**ABSTRACT**  
Recent advances in Unmanned Aerial Vehicles (UAVs) have resulted in their quick adoption for wide a range of civilian applications, including precision agriculture, biosecurity, disaster monitoring and surveillance. UAVs offer low-cost platforms with flexible hardware configurations, as well as an increasing number of autonomous capabilities, including take-off, landing, object tracking and obstacle avoidance. However, little attention has been paid to how UAVs deal with object detection uncertainties caused by false readings from vision-based detectors, data noise, vibrations, and occlusion. In most situations, the relevance and understanding of these detections are delegated to human operators, as many UAVs have limited cognition power to interact autonomously with the environment. This paper presents a framework for autonomous navigation under uncertainty in outdoor scenarios for small UAVs using a probabilistic-based motion planner. The framework is evaluated with real flight tests using a sub 2 kg quadrotor UAV and illustrated in victim finding Search and Rescue (SAR) case study in a forest/bushland. The navigation problem is modelled using a Partially Observable Markov Decision Process (POMDP), and solved in real time onboard the small UAV using Augmented Belief Trees (ABT) and the TAPIR toolkit. Results from experiments using colour and thermal imagery show that the proposed motion planner provides accurate victim localisation coordinates, as the UAV has the flexibility to interact with the environment and obtain clearer visualisations of any potential victims compared to the baseline motion planner. Incorporating this system allows optimised UAV surveillance operations by diminishing false positive readings from vision-based object detectors.

{{</citation>}}


## cs.SD (2)



### (3/125) PMVC: Data Augmentation-Based Prosody Modeling for Expressive Voice Conversion (Yimin Deng et al., 2023)

{{<citation>}}

Yimin Deng, Huaizhen Tang, Xulong Zhang, Jianzong Wang, Ning Cheng, Jing Xiao. (2023)  
**PMVC: Data Augmentation-Based Prosody Modeling for Expressive Voice Conversion**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: AI, Augmentation  
[Paper Link](http://arxiv.org/abs/2308.11084v1)  

---


**ABSTRACT**  
Voice conversion as the style transfer task applied to speech, refers to converting one person's speech into a new speech that sounds like another person's. Up to now, there has been a lot of research devoted to better implementation of VC tasks. However, a good voice conversion model should not only match the timbre information of the target speaker, but also expressive information such as prosody, pace, pause, etc. In this context, prosody modeling is crucial for achieving expressive voice conversion that sounds natural and convincing. Unfortunately, prosody modeling is important but challenging, especially without text transcriptions. In this paper, we firstly propose a novel voice conversion framework named 'PMVC', which effectively separates and models the content, timbre, and prosodic information from the speech without text transcriptions. Specially, we introduce a new speech augmentation algorithm for robust prosody extraction. And building upon this, mask and predict mechanism is applied in the disentanglement of prosody and content information. The experimental results on the AIShell-3 corpus supports our improvement of naturalness and similarity of converted speech.

{{</citation>}}


### (4/125) TokenSplit: Using Discrete Speech Representations for Direct, Refined, and Transcript-Conditioned Speech Separation and Recognition (Hakan Erdogan et al., 2023)

{{<citation>}}

Hakan Erdogan, Scott Wisdom, Xuankai Chang, Zalán Borsos, Marco Tagliasacchi, Neil Zeghidour, John R. Hershey. (2023)  
**TokenSplit: Using Discrete Speech Representations for Direct, Refined, and Transcript-Conditioned Speech Separation and Recognition**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10415v1)  

---


**ABSTRACT**  
We present TokenSplit, a speech separation model that acts on discrete token sequences. The model is trained on multiple tasks simultaneously: separate and transcribe each speech source, and generate speech from text. The model operates on transcripts and audio token sequences and achieves multiple tasks through masking of inputs. The model is a sequence-to-sequence encoder-decoder model that uses the Transformer architecture. We also present a "refinement" version of the model that predicts enhanced audio tokens from the audio tokens of speech separated by a conventional separation model. Using both objective metrics and subjective MUSHRA listening tests, we show that our model achieves excellent performance in terms of separation, both with or without transcript conditioning. We also measure the automatic speech recognition (ASR) performance and provide audio samples of speech synthesis to demonstrate the additional utility of our model.

{{</citation>}}


## cs.SE (5)



### (5/125) PrAIoritize: Learning to Prioritize Smart Contract Bugs and Vulnerabilities (Majd Soud et al., 2023)

{{<citation>}}

Majd Soud, Grischa Liebel, Mohammad Hamdaqa. (2023)  
**PrAIoritize: Learning to Prioritize Smart Contract Bugs and Vulnerabilities**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, BERT, LSTM  
[Paper Link](http://arxiv.org/abs/2308.11082v1)  

---


**ABSTRACT**  
Smart contract vulnerabilities and bugs have become a key concern for software engineers, as they can lead to significant financial losses, reputational damage, and legal issues. Therefore, prioritizing bug fixing for smart contracts is critical to maintaining trust. Due to the lack of tracking tools, prioritizing smart contract-reported bugs is done manually, which is a tedious task, limits bug triaging, and needs specialized knowledge. Towards this end, we propose PrAIoritize; an automated approach for predicting smart contract bug priorities that assist software engineers in prioritizing highly urgent bug reports. PrAIoritize consists of two main phases: 1) automatic labeling, which involves the automatic construction of a smart contract keyword lexicon and the automatic assignment of priority levels to unlabeled bug reports; 2) model construction, which involves feature engineering and designs layers of feed-forward neural networks (FFNNs) and bidirectional long short-term memory (BiLSTM) with multi-class classification to better capture the features of the textual descriptions of bugs and predict their priority levels. The model then is trained using smart contract bug reports collected from two data sources: open-source software (OSS) projects available on GitHub and NVD vulnerability database. Our evaluation demonstrates significant improvement over state-of-the-art baselines and commonly used pre-trained models (e.g. BERT) for similar classification tasks, with 5.75%-35.29% increase in F-measure, precision, and recall.

{{</citation>}}


### (6/125) Large Language Models for Software Engineering: A Systematic Literature Review (Xinyi Hou et al., 2023)

{{<citation>}}

Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, Haoyu Wang. (2023)  
**Large Language Models for Software Engineering: A Systematic Literature Review**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10620v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have significantly impacted numerous domains, notably including Software Engineering (SE). Nevertheless, a well-rounded understanding of the application, effects, and possible limitations of LLMs within SE is still in its early stages. To bridge this gap, our systematic literature review takes a deep dive into the intersection of LLMs and SE, with a particular focus on understanding how LLMs can be exploited in SE to optimize processes and outcomes. Through a comprehensive review approach, we collect and analyze a total of 229 research papers from 2017 to 2023 to answer four key research questions (RQs). In RQ1, we categorize and provide a comparative analysis of different LLMs that have been employed in SE tasks, laying out their distinctive features and uses. For RQ2, we detail the methods involved in data collection, preprocessing, and application in this realm, shedding light on the critical role of robust, well-curated datasets for successful LLM implementation. RQ3 allows us to examine the specific SE tasks where LLMs have shown remarkable success, illuminating their practical contributions to the field. Finally, RQ4 investigates the strategies employed to optimize and evaluate the performance of LLMs in SE, as well as the common techniques related to prompt optimization. Armed with insights drawn from addressing the aforementioned RQs, we sketch a picture of the current state-of-the-art, pinpointing trends, identifying gaps in existing research, and flagging promising areas for future study.

{{</citation>}}


### (7/125) Incorprating Prompt tuning for Commit classification with prior Knowledge (Jiajun Tong et al., 2023)

{{<citation>}}

Jiajun Tong, Zhixiao Wang, Xiaobin Rui. (2023)  
**Incorprating Prompt tuning for Commit classification with prior Knowledge**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2308.10576v1)  

---


**ABSTRACT**  
Commit Classification(CC) is an important task in software maintenance since it helps software developers classify code changes into different types according to their nature and purpose. This allows them to better understand how their development efforts are progressing, identify areas where they need improvement. However, existing methods are all discriminative models, usually with complex architectures that require additional output layers to produce class label probabilities. Moreover, they require a large amount of labeled data for fine-tuning, and it is difficult to learn effective classification boundaries in the case of limited labeled data. To solve above problems, we propose a generative framework that Incorporating prompt-tuning for commit classification with prior knowledge (IPCK) https://github.com/AppleMax1992/IPCK, which simplifies the model structure and learns features across different tasks. It can still reach the SOTA performance with only limited samples. Firstly, we proposed a generative framework based on T5. This encoder-decoder construction method unifies different CC task into a text2text problem, which simplifies the structure of the model by not requiring an extra output layer. Second, instead of fine-tuning, we design an prompt-tuning solution which can be adopted in few-shot scenarios with only limit samples. Furthermore, we incorporate prior knowledge via an external knowledge graph to map the probabilities of words into the final labels in the speech machine step to improve performance in few-shot scenarios. Extensive experiments on two open available datasets show that our framework can solve the CC problem simply but effectively in few-shot and zeroshot scenarios, while improving the adaptability of the model without requiring a large amount of training samples for fine-tuning.

{{</citation>}}


### (8/125) When Less is Enough: Positive and Unlabeled Learning Model for Vulnerability Detection (Xin-Cheng Wen et al., 2023)

{{<citation>}}

Xin-Cheng Wen, Xinchen Wang, Cuiyun Gao, Shaohua Wang, Yang Liu, Zhaoquan Gu. (2023)  
**When Less is Enough: Positive and Unlabeled Learning Model for Vulnerability Detection**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CR, cs-SE, cs.SE  
Keywords: Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2308.10523v1)  

---


**ABSTRACT**  
Automated code vulnerability detection has gained increasing attention in recent years. The deep learning (DL)-based methods, which implicitly learn vulnerable code patterns, have proven effective in vulnerability detection. The performance of DL-based methods usually relies on the quantity and quality of labeled data. However, the current labeled data are generally automatically collected, such as crawled from human-generated commits, making it hard to ensure the quality of the labels. Prior studies have demonstrated that the non-vulnerable code (i.e., negative labels) tends to be unreliable in commonly-used datasets, while vulnerable code (i.e., positive labels) is more determined. Considering the large numbers of unlabeled data in practice, it is necessary and worth exploring to leverage the positive data and large numbers of unlabeled data for more accurate vulnerability detection.   In this paper, we focus on the Positive and Unlabeled (PU) learning problem for vulnerability detection and propose a novel model named PILOT, i.e., PositIve and unlabeled Learning mOdel for vulnerability deTection. PILOT only learns from positive and unlabeled data for vulnerability detection. It mainly contains two modules: (1) A distance-aware label selection module, aiming at generating pseudo-labels for selected unlabeled data, which involves the inter-class distance prototype and progressive fine-tuning; (2) A mixed-supervision representation learning module to further alleviate the influence of noise and enhance the discrimination of representations.

{{</citation>}}


### (9/125) Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models (Martin Weyssow et al., 2023)

{{<citation>}}

Martin Weyssow, Xin Zhou, Kisub Kim, David Lo, Houari Sahraoui. (2023)  
**Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10462v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) possess impressive capabilities to generate meaningful code snippets given natural language intents in zero-shot, i.e., without the need for specific fine-tuning. In the perspective of unleashing their full potential, prior work has demonstrated the benefits of fine-tuning the models to task-specific data. However, fine-tuning process demands heavy computational costs and is intractable when resources are scarce, especially for models with billions of parameters. In light of these challenges, previous studies explored In-Context Learning (ICL) as an effective strategy to generate contextually appropriate code without fine-tuning. However, it operates at inference time and does not involve learning task-specific parameters, potentially limiting the model's performance on downstream tasks. In this context, we foresee that Parameter-Efficient Fine-Tuning (PEFT) techniques carry a high potential for efficiently specializing LLMs to task-specific data. In this paper, we deliver a comprehensive study of LLMs with the impact of PEFT techniques under the automated code generation scenario. Our experimental results reveal the superiority and potential of such techniques over ICL on a wide range of LLMs in reducing the computational burden and improving performance. Therefore, the study opens opportunities for broader applications of PEFT in software engineering scenarios.

{{</citation>}}


## cs.CV (38)



### (10/125) Long-Term Prediction of Natural Video Sequences with Robust Video Predictors (Luke Ditria et al., 2023)

{{<citation>}}

Luke Ditria, Tom Drummond. (2023)  
**Long-Term Prediction of Natural Video Sequences with Robust Video Predictors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.11079v1)  

---


**ABSTRACT**  
Predicting high dimensional video sequences is a curiously difficult problem. The number of possible futures for a given video sequence grows exponentially over time due to uncertainty. This is especially evident when trying to predict complicated natural video scenes from a limited snapshot of the world. The inherent uncertainty accumulates the further into the future you predict making long-term prediction very difficult. In this work we introduce a number of improvements to existing work that aid in creating Robust Video Predictors (RoViPs). We show that with a combination of deep Perceptual and uncertainty-based reconstruction losses we are able to create high quality short-term predictions. Attention-based skip connections are utilised to allow for long range spatial movement of input features to further improve performance. Finally, we show that by simply making the predictor robust to its own prediction errors, it is possible to produce very long, realistic natural video sequences using an iterated single-step prediction task.

{{</citation>}}


### (11/125) Audio-Visual Class-Incremental Learning (Weiguo Pian et al., 2023)

{{<citation>}}

Weiguo Pian, Shentong Mo, Yunhui Guo, Yapeng Tian. (2023)  
**Audio-Visual Class-Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.11073v1)  

---


**ABSTRACT**  
In this paper, we introduce audio-visual class-incremental learning, a class-incremental learning scenario for audio-visual video recognition. We demonstrate that joint audio-visual modeling can improve class-incremental learning, but current methods fail to preserve semantic similarity between audio and visual features as incremental step grows. Furthermore, we observe that audio-visual correlations learned in previous tasks can be forgotten as incremental steps progress, leading to poor performance. To overcome these challenges, we propose AV-CIL, which incorporates Dual-Audio-Visual Similarity Constraint (D-AVSC) to maintain both instance-aware and class-aware semantic similarity between audio-visual modalities and Visual Attention Distillation (VAD) to retain previously learned audio-guided visual attentive ability. We create three audio-visual class-incremental datasets, AVE-Class-Incremental (AVE-CI), Kinetics-Sounds-Class-Incremental (K-S-CI), and VGGSound100-Class-Incremental (VS100-CI) based on the AVE, Kinetics-Sounds, and VGGSound datasets, respectively. Our experiments on AVE-CI, K-S-CI, and VS100-CI demonstrate that AV-CIL significantly outperforms existing class-incremental learning methods in audio-visual class-incremental learning. Code and data are available at: https://github.com/weiguoPian/AV-CIL_ICCV2023.

{{</citation>}}


### (12/125) TeD-SPAD: Temporal Distinctiveness for Self-supervised Privacy-preservation for video Anomaly Detection (Joseph Fioresi et al., 2023)

{{<citation>}}

Joseph Fioresi, Ishan Rajendrakumar Dave, Mubarak Shah. (2023)  
**TeD-SPAD: Temporal Distinctiveness for Self-supervised Privacy-preservation for video Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.11072v1)  

---


**ABSTRACT**  
Video anomaly detection (VAD) without human monitoring is a complex computer vision task that can have a positive impact on society if implemented successfully. While recent advances have made significant progress in solving this task, most existing approaches overlook a critical real-world concern: privacy. With the increasing popularity of artificial intelligence technologies, it becomes crucial to implement proper AI ethics into their development. Privacy leakage in VAD allows models to pick up and amplify unnecessary biases related to people's personal information, which may lead to undesirable decision making. In this paper, we propose TeD-SPAD, a privacy-aware video anomaly detection framework that destroys visual private information in a self-supervised manner. In particular, we propose the use of a temporally-distinct triplet loss to promote temporally discriminative features, which complements current weakly-supervised VAD methods. Using TeD-SPAD, we achieve a positive trade-off between privacy protection and utility anomaly detection performance on three popular weakly supervised VAD datasets: UCF-Crime, XD-Violence, and ShanghaiTech. Our proposed anonymization model reduces private attribute prediction by 32.25% while only reducing frame-level ROC AUC on the UCF-Crime anomaly detection dataset by 3.69%. Project Page: https://joefioresi718.github.io/TeD-SPAD_webpage/

{{</citation>}}


### (13/125) Beyond Discriminative Regions: Saliency Maps as Alternatives to CAMs for Weakly Supervised Semantic Segmentation (M. Maruf et al., 2023)

{{<citation>}}

M. Maruf, Arka Daw, Amartya Dutta, Jie Bu, Anuj Karpatne. (2023)  
**Beyond Discriminative Regions: Saliency Maps as Alternatives to CAMs for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.11052v1)  

---


**ABSTRACT**  
In recent years, several Weakly Supervised Semantic Segmentation (WS3) methods have been proposed that use class activation maps (CAMs) generated by a classifier to produce pseudo-ground truths for training segmentation models. While CAMs are good at highlighting discriminative regions (DR) of an image, they are known to disregard regions of the object that do not contribute to the classifier's prediction, termed non-discriminative regions (NDR). In contrast, attribution methods such as saliency maps provide an alternative approach for assigning a score to every pixel based on its contribution to the classification prediction. This paper provides a comprehensive comparison between saliencies and CAMs for WS3. Our study includes multiple perspectives on understanding their similarities and dissimilarities. Moreover, we provide new evaluation metrics that perform a comprehensive assessment of WS3 performance of alternative methods w.r.t. CAMs. We demonstrate the effectiveness of saliencies in addressing the limitation of CAMs through our empirical studies on benchmark datasets. Furthermore, we propose random cropping as a stochastic aggregation technique that improves the performance of saliency, making it a strong alternative to CAM for WS3.

{{</citation>}}


### (14/125) Spectral Graphormer: Spectral Graph-based Transformer for Egocentric Two-Hand Reconstruction using Multi-View Color Images (Tze Ho Elden Tse et al., 2023)

{{<citation>}}

Tze Ho Elden Tse, Franziska Mueller, Zhengyang Shen, Danhang Tang, Thabo Beeler, Mingsong Dou, Yinda Zhang, Sasa Petrovic, Hyung Jin Chang, Jonathan Taylor, Bardia Doosti. (2023)  
**Spectral Graphormer: Spectral Graph-based Transformer for Egocentric Two-Hand Reconstruction using Multi-View Color Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11015v1)  

---


**ABSTRACT**  
We propose a novel transformer-based framework that reconstructs two high fidelity hands from multi-view RGB images. Unlike existing hand pose estimation methods, where one typically trains a deep network to regress hand model parameters from single RGB image, we consider a more challenging problem setting where we directly regress the absolute root poses of two-hands with extended forearm at high resolution from egocentric view. As existing datasets are either infeasible for egocentric viewpoints or lack background variations, we create a large-scale synthetic dataset with diverse scenarios and collect a real dataset from multi-calibrated camera setup to verify our proposed multi-view image feature fusion strategy. To make the reconstruction physically plausible, we propose two strategies: (i) a coarse-to-fine spectral graph convolution decoder to smoothen the meshes during upsampling and (ii) an optimisation-based refinement stage at inference to prevent self-penetrations. Through extensive quantitative and qualitative evaluations, we show that our framework is able to produce realistic two-hand reconstructions and demonstrate the generalisation of synthetic-trained models to real data, as well as real-time AR/VR applications.

{{</citation>}}


### (15/125) VQA Therapy: Exploring Answer Differences by Visually Grounding Answers (Chongyan Chen et al., 2023)

{{<citation>}}

Chongyan Chen, Samreen Anjum, Danna Gurari. (2023)  
**VQA Therapy: Exploring Answer Differences by Visually Grounding Answers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.11662v1)  

---


**ABSTRACT**  
Visual question answering is a task of predicting the answer to a question about an image. Given that different people can provide different answers to a visual question, we aim to better understand why with answer groundings. We introduce the first dataset that visually grounds each unique answer to each visual question, which we call VQAAnswerTherapy. We then propose two novel problems of predicting whether a visual question has a single answer grounding and localizing all answer groundings. We benchmark modern algorithms for these novel problems to show where they succeed and struggle. The dataset and evaluation server can be found publicly at https://vizwiz.org/tasks-and-datasets/vqa-answer-therapy/.

{{</citation>}}


### (16/125) SupEuclid: Extremely Simple, High Quality OoD Detection with Supervised Contrastive Learning and Euclidean Distance (Jarrod Haas, 2023)

{{<citation>}}

Jarrod Haas. (2023)  
**SupEuclid: Extremely Simple, High Quality OoD Detection with Supervised Contrastive Learning and Euclidean Distance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.10973v1)  

---


**ABSTRACT**  
Out-of-Distribution (OoD) detection has developed substantially in the past few years, with available methods approaching, and in a few cases achieving, perfect data separation on standard benchmarks. These results generally involve large or complex models, pretraining, exposure to OoD examples or extra hyperparameter tuning. Remarkably, it is possible to achieve results that can exceed many of these state-of-the-art methods with a very simple method. We demonstrate that ResNet18 trained with Supervised Contrastive Learning (SCL) produces state-of-the-art results out-of-the-box on near and far OoD detection benchmarks using only Euclidean distance as a scoring rule. This may obviate the need in some cases for more sophisticated methods or larger models, and at the very least provides a very strong, easy to use baseline for further experimentation and analysis.

{{</citation>}}


### (17/125) MRI Field-transfer Reconstruction with Limited Data: Regularization by Neural Style Transfer (Guoyao Shen et al., 2023)

{{<citation>}}

Guoyao Shen, Yancheng Zhu, Hernan Jara, Sean B. Andersson, Chad W. Farris, Stephan Anderson, Xin Zhang. (2023)  
**MRI Field-transfer Reconstruction with Limited Data: Regularization by Neural Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, physics-med-ph  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2308.10968v1)  

---


**ABSTRACT**  
Recent works have demonstrated success in MRI reconstruction using deep learning-based models. However, most reported approaches require training on a task-specific, large-scale dataset. Regularization by denoising (RED) is a general pipeline which embeds a denoiser as a prior for image reconstruction. The potential of RED has been demonstrated for multiple image-related tasks such as denoising, deblurring and super-resolution. In this work, we propose a regularization by neural style transfer (RNST) method to further leverage the priors from the neural transfer and denoising engine. This enables RNST to reconstruct a high-quality image from a noisy low-quality image with different image styles and limited data. We validate RNST with clinical MRI scans from 1.5T and 3T and show that RNST can significantly boost image quality. Our results highlight the capability of the RNST framework for MRI reconstruction and the potential for reconstruction tasks with limited data.

{{</citation>}}


### (18/125) Few-Shot Physically-Aware Articulated Mesh Generation via Hierarchical Deformation (Xueyi Liu et al., 2023)

{{<citation>}}

Xueyi Liu, Bin Wang, He Wang, Li Yi. (2023)  
**Few-Shot Physically-Aware Articulated Mesh Generation via Hierarchical Deformation**  

---
Primary Category: cs.CV  
Categories: cs-CG, cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.10898v1)  

---


**ABSTRACT**  
We study the problem of few-shot physically-aware articulated mesh generation. By observing an articulated object dataset containing only a few examples, we wish to learn a model that can generate diverse meshes with high visual fidelity and physical validity. Previous mesh generative models either have difficulties in depicting a diverse data space from only a few examples or fail to ensure physical validity of their samples. Regarding the above challenges, we propose two key innovations, including 1) a hierarchical mesh deformation-based generative model based upon the divide-and-conquer philosophy to alleviate the few-shot challenge by borrowing transferrable deformation patterns from large scale rigid meshes and 2) a physics-aware deformation correction scheme to encourage physically plausible generations. We conduct extensive experiments on 6 articulated categories to demonstrate the superiority of our method in generating articulated meshes with better diversity, higher visual fidelity, and better physical validity over previous methods in the few-shot setting. Further, we validate solid contributions of our two innovations in the ablation study. Project page with code is available at https://meowuu7.github.io/few-arti-obj-gen.

{{</citation>}}


### (19/125) Can Language Models Learn to Listen? (Evonne Ng et al., 2023)

{{<citation>}}

Evonne Ng, Sanjay Subramanian, Dan Klein, Angjoo Kanazawa, Trevor Darrell, Shiry Ginosar. (2023)  
**Can Language Models Learn to Listen?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10897v1)  

---


**ABSTRACT**  
We present a framework for generating appropriate facial responses from a listener in dyadic social interactions based on the speaker's words. Given an input transcription of the speaker's words with their timestamps, our approach autoregressively predicts a response of a listener: a sequence of listener facial gestures, quantized using a VQ-VAE. Since gesture is a language component, we propose treating the quantized atomic motion elements as additional language token inputs to a transformer-based large language model. Initializing our transformer with the weights of a language model pre-trained only on text results in significantly higher quality listener responses than training a transformer from scratch. We show that our generated listener motion is fluent and reflective of language semantics through quantitative metrics and a qualitative user study. In our evaluation, we analyze the model's ability to utilize temporal and semantic aspects of spoken text. Project page: https://people.eecs.berkeley.edu/~evonne_ng/projects/text2listen/

{{</citation>}}


### (20/125) Vision Transformer Pruning Via Matrix Decomposition (Tianyi Sun, 2023)

{{<citation>}}

Tianyi Sun. (2023)  
**Vision Transformer Pruning Via Matrix Decomposition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-CO  
Keywords: Pruning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.10839v1)  

---


**ABSTRACT**  
This is a further development of Vision Transformer Pruning via matrix decomposition. The purpose of the Vision Transformer Pruning is to prune the dimension of the linear projection of the dataset by learning their associated importance score in order to reduce the storage, run-time memory, and computational demands. In this paper we further reduce dimension and complexity of the linear projection by implementing and comparing several matrix decomposition methods while preserving the generated important features. We end up selected the Singular Value Decomposition as the method to achieve our goal by comparing the original accuracy scores in the original Github repository and the accuracy scores of using those matrix decomposition methods, including Singular Value Decomposition, four versions of QR Decomposition, and LU factorization.

{{</citation>}}


### (21/125) Pixel Adaptive Deep Unfolding Transformer for Hyperspectral Image Reconstruction (Miaoyu Li et al., 2023)

{{<citation>}}

Miaoyu Li, Ying Fu, Ji Liu, Yulun Zhang. (2023)  
**Pixel Adaptive Deep Unfolding Transformer for Hyperspectral Image Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10820v1)  

---


**ABSTRACT**  
Hyperspectral Image (HSI) reconstruction has made gratifying progress with the deep unfolding framework by formulating the problem into a data module and a prior module. Nevertheless, existing methods still face the problem of insufficient matching with HSI data. The issues lie in three aspects: 1) fixed gradient descent step in the data module while the degradation of HSI is agnostic in the pixel-level. 2) inadequate prior module for 3D HSI cube. 3) stage interaction ignoring the differences in features at different stages. To address these issues, in this work, we propose a Pixel Adaptive Deep Unfolding Transformer (PADUT) for HSI reconstruction. In the data module, a pixel adaptive descent step is employed to focus on pixel-level agnostic degradation. In the prior module, we introduce the Non-local Spectral Transformer (NST) to emphasize the 3D characteristics of HSI for recovering. Moreover, inspired by the diverse expression of features in different stages and depths, the stage interaction is improved by the Fast Fourier Transform (FFT). Experimental results on both simulated and real scenes exhibit the superior performance of our method compared to state-of-the-art HSI reconstruction methods. The code is released at: https://github.com/MyuLi/PADUT.

{{</citation>}}


### (22/125) Jumping through Local Minima: Quantization in the Loss Landscape of Vision Transformers (Natalia Frumkin et al., 2023)

{{<citation>}}

Natalia Frumkin, Dibakar Gope, Diana Marculescu. (2023)  
**Jumping through Local Minima: Quantization in the Loss Landscape of Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10814v1)  

---


**ABSTRACT**  
Quantization scale and bit-width are the most important parameters when considering how to quantize a neural network. Prior work focuses on optimizing quantization scales in a global manner through gradient methods (gradient descent \& Hessian analysis). Yet, when applying perturbations to quantization scales, we observe a very jagged, highly non-smooth test loss landscape. In fact, small perturbations in quantization scale can greatly affect accuracy, yielding a $0.5-0.8\%$ accuracy boost in 4-bit quantized vision transformers (ViTs). In this regime, gradient methods break down, since they cannot reliably reach local minima. In our work, dubbed Evol-Q, we use evolutionary search to effectively traverse the non-smooth landscape. Additionally, we propose using an infoNCE loss, which not only helps combat overfitting on the small calibration dataset ($1,000$ images) but also makes traversing such a highly non-smooth surface easier. Evol-Q improves the top-1 accuracy of a fully quantized ViT-Base by $10.30\%$, $0.78\%$, and $0.15\%$ for $3$-bit, $4$-bit, and $8$-bit weight quantization levels. Extensive experiments on a variety of CNN and ViT architectures further demonstrate its robustness in extreme quantization scenarios. Our code is available at https://github.com/enyac-group/evol-q

{{</citation>}}


### (23/125) CoNe: Contrast Your Neighbours for Supervised Image Classification (Mingkai Zheng et al., 2023)

{{<citation>}}

Mingkai Zheng, Shan You, Lang Huang, Xiu Su, Fei Wang, Chen Qian, Xiaogang Wang, Chang Xu. (2023)  
**CoNe: Contrast Your Neighbours for Supervised Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.10761v1)  

---


**ABSTRACT**  
Image classification is a longstanding problem in computer vision and machine learning research. Most recent works (e.g. SupCon , Triplet, and max-margin) mainly focus on grouping the intra-class samples aggressively and compactly, with the assumption that all intra-class samples should be pulled tightly towards their class centers. However, such an objective will be very hard to achieve since it ignores the intra-class variance in the dataset. (i.e. different instances from the same class can have significant differences). Thus, such a monotonous objective is not sufficient. To provide a more informative objective, we introduce Contrast Your Neighbours (CoNe) - a simple yet practical learning framework for supervised image classification. Specifically, in CoNe, each sample is not only supervised by its class center but also directly employs the features of its similar neighbors as anchors to generate more adaptive and refined targets. Moreover, to further boost the performance, we propose ``distributional consistency" as a more informative regularization to enable similar instances to have a similar probability distribution. Extensive experimental results demonstrate that CoNe achieves state-of-the-art performance across different benchmark datasets, network architectures, and settings. Notably, even without a complicated training recipe, our CoNe achieves 80.8\% Top-1 accuracy on ImageNet with ResNet-50, which surpasses the recent Timm training recipe (80.4\%). Code and pre-trained models are available at \href{https://github.com/mingkai-zheng/CoNe}{https://github.com/mingkai-zheng/CoNe}.

{{</citation>}}


### (24/125) Boosting Adversarial Attack with Similar Target (Shuo Zhang et al., 2023)

{{<citation>}}

Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen. (2023)  
**Boosting Adversarial Attack with Similar Target**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Adversarial Attack, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.10743v1)  

---


**ABSTRACT**  
Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.

{{</citation>}}


### (25/125) Patch Is Not All You Need (Changzhen Li et al., 2023)

{{<citation>}}

Changzhen Li, Jie Zhang, Yang Wei, Zhilong Ji, Jinfeng Bai, Shiguang Shan. (2023)  
**Patch Is Not All You Need**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10729v1)  

---


**ABSTRACT**  
Vision Transformers have achieved great success in computer visions, delivering exceptional performance across various tasks. However, their inherent reliance on sequential input enforces the manual partitioning of images into patch sequences, which disrupts the image's inherent structural and semantic continuity. To handle this, we propose a novel Pattern Transformer (Patternformer) to adaptively convert images to pattern sequences for Transformer input. Specifically, we employ the Convolutional Neural Network to extract various patterns from the input image, with each channel representing a unique pattern that is fed into the succeeding Transformer as a visual token. By enabling the network to optimize these patterns, each pattern concentrates on its local region of interest, thereby preserving its intrinsic structural and semantic information. Only employing the vanilla ResNet and Transformer, we have accomplished state-of-the-art performance on CIFAR-10 and CIFAR-100, and have achieved competitive results on ImageNet.

{{</citation>}}


### (26/125) Test-time augmentation-based active learning and self-training for label-efficient segmentation (Bella Specktor-Fadida et al., 2023)

{{<citation>}}

Bella Specktor-Fadida, Anna Levchakov, Dana Schonberger, Liat Ben-Sira, Dafna Ben-Bashat, Leo Joskowicz. (2023)  
**Test-time augmentation-based active learning and self-training for label-efficient segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.10727v1)  

---


**ABSTRACT**  
Deep learning techniques depend on large datasets whose annotation is time-consuming. To reduce annotation burden, the self-training (ST) and active-learning (AL) methods have been developed as well as methods that combine them in an iterative fashion. However, it remains unclear when each method is the most useful, and when it is advantageous to combine them. In this paper, we propose a new method that combines ST with AL using Test-Time Augmentations (TTA). First, TTA is performed on an initial teacher network. Then, cases for annotation are selected based on the lowest estimated Dice score. Cases with high estimated scores are used as soft pseudo-labels for ST. The selected annotated cases are trained with existing annotated cases and ST cases with border slices annotations. We demonstrate the method on MRI fetal body and placenta segmentation tasks with different data variability characteristics. Our results indicate that ST is highly effective for both tasks, boosting performance for in-distribution (ID) and out-of-distribution (OOD) data. However, while self-training improved the performance of single-sequence fetal body segmentation when combined with AL, it slightly deteriorated performance of multi-sequence placenta segmentation on ID data. AL was helpful for the high variability placenta data, but did not improve upon random selection for the single-sequence body data. For fetal body segmentation sequence transfer, combining AL with ST following ST iteration yielded a Dice of 0.961 with only 6 original scans and 2 new sequence scans. Results using only 15 high-variability placenta cases were similar to those using 50 cases. Code is available at: https://github.com/Bella31/TTA-quality-estimation-ST-AL

{{</citation>}}


### (27/125) Co-Speech Gesture Detection through Multi-phase Sequence Labeling (Esam Ghaleb et al., 2023)

{{<citation>}}

Esam Ghaleb, Ilya Burenko, Marlou Rasenberg, Wim Pouw, Peter Uhrig, Judith Holler, Ivan Toni, Aslı Özyürek, Raquel Fernández. (2023)  
**Co-Speech Gesture Detection through Multi-phase Sequence Labeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10680v1)  

---


**ABSTRACT**  
Gestures are integral components of face-to-face communication. They unfold over time, often following predictable movement phases of preparation, stroke, and retraction. Yet, the prevalent approach to automatic gesture detection treats the problem as binary classification, classifying a segment as either containing a gesture or not, thus failing to capture its inherently sequential and contextual nature. To address this, we introduce a novel framework that reframes the task as a multi-phase sequence labeling problem rather than binary classification. Our model processes sequences of skeletal movements over time windows, uses Transformer encoders to learn contextual embeddings, and leverages Conditional Random Fields to perform sequence labeling. We evaluate our proposal on a large dataset of diverse co-speech gestures in task-oriented face-to-face dialogues. The results consistently demonstrate that our method significantly outperforms strong baseline models in detecting gesture strokes. Furthermore, applying Transformer encoders to learn contextual embeddings from movement sequences substantially improves gesture unit detection. These results highlight our framework's capacity to capture the fine-grained dynamics of co-speech gesture phases, paving the way for more nuanced and accurate gesture detection and analysis.

{{</citation>}}


### (28/125) bbOCR: An Open-source Multi-domain OCR Pipeline for Bengali Documents (Imam Mohammad Zulkarnain et al., 2023)

{{<citation>}}

Imam Mohammad Zulkarnain, Shayekh Bin Islam, Md. Zami Al Zunaed Farabe, Md. Mehedi Hasan Shawon, Jawaril Munshad Abedin, Beig Rajibul Hasan, Marsia Haque, Istiak Shihab, Syed Mobassir, MD. Nazmuddoha Ansary, Asif Sushmit, Farig Sadeque. (2023)  
**bbOCR: An Open-source Multi-domain OCR Pipeline for Bengali Documents**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, OCR  
[Paper Link](http://arxiv.org/abs/2308.10647v2)  

---


**ABSTRACT**  
Despite the existence of numerous Optical Character Recognition (OCR) tools, the lack of comprehensive open-source systems hampers the progress of document digitization in various low-resource languages, including Bengali. Low-resource languages, especially those with an alphasyllabary writing system, suffer from the lack of large-scale datasets for various document OCR components such as word-level OCR, document layout extraction, and distortion correction; which are available as individual modules in high-resource languages. In this paper, we introduce Bengali$.$AI-BRACU-OCR (bbOCR): an open-source scalable document OCR system that can reconstruct Bengali documents into a structured searchable digitized format that leverages a novel Bengali text recognition model and two novel synthetic datasets. We present extensive component-level and system-level evaluation: both use a novel diversified evaluation dataset and comprehensive evaluation metrics. Our extensive evaluation suggests that our proposed solution is preferable over the current state-of-the-art Bengali OCR systems. The source codes and datasets are available here: https://bengaliai.github.io/bbocr.

{{</citation>}}


### (29/125) GaitPT: Skeletons Are All You Need For Gait Recognition (Andy Catruna et al., 2023)

{{<citation>}}

Andy Catruna, Adrian Cosma, Emilian Radoi. (2023)  
**GaitPT: Skeletons Are All You Need For Gait Recognition**  

---
Primary Category: cs.CV  
Categories: 68U10, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10623v1)  

---


**ABSTRACT**  
The analysis of patterns of walking is an important area of research that has numerous applications in security, healthcare, sports and human-computer interaction. Lately, walking patterns have been regarded as a unique fingerprinting method for automatic person identification at a distance. In this work, we propose a novel gait recognition architecture called Gait Pyramid Transformer (GaitPT) that leverages pose estimation skeletons to capture unique walking patterns, without relying on appearance information. GaitPT adopts a hierarchical transformer architecture that effectively extracts both spatial and temporal features of movement in an anatomically consistent manner, guided by the structure of the human skeleton. Our results show that GaitPT achieves state-of-the-art performance compared to other skeleton-based gait recognition works, in both controlled and in-the-wild scenarios. GaitPT obtains 82.6% average accuracy on CASIA-B, surpassing other works by a margin of 6%. Moreover, it obtains 52.16% Rank-1 accuracy on GREW, outperforming both skeleton-based and appearance-based approaches.

{{</citation>}}


### (30/125) Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer (Zhijin Ge et al., 2023)

{{<citation>}}

Zhijin Ge, Fanhua Shang, Hongying Liu, Yuanyuan Liu, Liang Wan, Wei Feng, Xiaosen Wang. (2023)  
**Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: ImageNet, Style Transfer  
[Paper Link](http://arxiv.org/abs/2308.10601v1)  

---


**ABSTRACT**  
Deep neural networks are vulnerable to adversarial examples crafted by applying human-imperceptible perturbations on clean inputs. Although many attack methods can achieve high success rates in the white-box setting, they also exhibit weak transferability in the black-box setting. Recently, various methods have been proposed to improve adversarial transferability, in which the input transformation is one of the most effective methods. In this work, we notice that existing input transformation-based works mainly adopt the transformed data in the same domain for augmentation. Inspired by domain generalization, we aim to further improve the transferability using the data augmented from different domains. Specifically, a style transfer network can alter the distribution of low-level visual features in an image while preserving semantic content for humans. Hence, we propose a novel attack method named Style Transfer Method (STM) that utilizes a proposed arbitrary style transfer network to transform the images into different domains. To avoid inconsistent semantic information of stylized images for the classification network, we fine-tune the style transfer network and mix up the generated images added by random noise with the original images to maintain semantic consistency and boost input diversity. Extensive experimental results on the ImageNet-compatible dataset show that our proposed method can significantly improve the adversarial transferability on either normally trained models or adversarially trained models than state-of-the-art input transformation-based attacks. Code is available at: https://github.com/Zhijin-Ge/STM.

{{</citation>}}


### (31/125) Image-free Classifier Injection for Zero-Shot Classification (Anders Christensen et al., 2023)

{{<citation>}}

Anders Christensen, Massimiliano Mancini, A. Sophia Koepke, Ole Winther, Zeynep Akata. (2023)  
**Image-free Classifier Injection for Zero-Shot Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.10599v1)  

---


**ABSTRACT**  
Zero-shot learning models achieve remarkable results on image classification for samples from classes that were not seen during training. However, such models must be trained from scratch with specialised methods: therefore, access to a training dataset is required when the need for zero-shot classification arises. In this paper, we aim to equip pre-trained models with zero-shot classification capabilities without the use of image data. We achieve this with our proposed Image-free Classifier Injection with Semantics (ICIS) that injects classifiers for new, unseen classes into pre-trained classification models in a post-hoc fashion without relying on image data. Instead, the existing classifier weights and simple class-wise descriptors, such as class names or attributes, are used. ICIS has two encoder-decoder networks that learn to reconstruct classifier weights from descriptors (and vice versa), exploiting (cross-)reconstruction and cosine losses to regularise the decoding process. Notably, ICIS can be cheaply trained and applied directly on top of pre-trained classification models. Experiments on benchmark ZSL datasets show that ICIS produces unseen classifier weights that achieve strong (generalised) zero-shot classification performance. Code is available at https://github.com/ExplainableML/ImageFreeZSL .

{{</citation>}}


### (32/125) CHORD: Category-level Hand-held Object Reconstruction via Shape Deformation (Kailin Li et al., 2023)

{{<citation>}}

Kailin Li, Lixin Yang, Haoyu Zhen, Zenan Lin, Xinyu Zhan, Licheng Zhong, Jian Xu, Kejian Wu, Cewu Lu. (2023)  
**CHORD: Category-level Hand-held Object Reconstruction via Shape Deformation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10574v1)  

---


**ABSTRACT**  
In daily life, humans utilize hands to manipulate objects. Modeling the shape of objects that are manipulated by the hand is essential for AI to comprehend daily tasks and to learn manipulation skills. However, previous approaches have encountered difficulties in reconstructing the precise shapes of hand-held objects, primarily owing to a deficiency in prior shape knowledge and inadequate data for training. As illustrated, given a particular type of tool, such as a mug, despite its infinite variations in shape and appearance, humans have a limited number of 'effective' modes and poses for its manipulation. This can be attributed to the fact that humans have mastered the shape prior of the 'mug' category, and can quickly establish the corresponding relations between different mug instances and the prior, such as where the rim and handle are located. In light of this, we propose a new method, CHORD, for Category-level Hand-held Object Reconstruction via shape Deformation. CHORD deforms a categorical shape prior for reconstructing the intra-class objects. To ensure accurate reconstruction, we empower CHORD with three types of awareness: appearance, shape, and interacting pose. In addition, we have constructed a new dataset, COMIC, of category-level hand-object interaction. COMIC contains a rich array of object instances, materials, hand interactions, and viewing directions. Extensive evaluation shows that CHORD outperforms state-of-the-art approaches in both quantitative and qualitative measures. Code, model, and datasets are available at https://kailinli.github.io/CHORD.

{{</citation>}}


### (33/125) Seeing the Intangible: Surveying Automatic High-Level Visual Understanding from Still Images (Delfina Sol Martinez Pandiani et al., 2023)

{{<citation>}}

Delfina Sol Martinez Pandiani, Valentina Presutti. (2023)  
**Seeing the Intangible: Surveying Automatic High-Level Visual Understanding from Still Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-CY, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.10562v1)  

---


**ABSTRACT**  
The field of Computer Vision (CV) was born with the single grand goal of complete image understanding: providing a complete semantic interpretation of an input image. What exactly this goal entails is not immediately straightforward, but theoretical hierarchies of visual understanding point towards a top level of full semantics, within which sits the most complex and subjective information humans can detect from visual data. In particular, non-concrete concepts including emotions, social values and ideologies seem to be protagonists of this "high-level" visual semantic understanding. While such "abstract concepts" are critical tools for image management and retrieval, their automatic recognition is still a challenge, exactly because they rest at the top of the "semantic pyramid": the well-known semantic gap problem is worsened given their lack of unique perceptual referents, and their reliance on more unspecific features than concrete concepts. Given that there seems to be very scarce explicit work within CV on the task of abstract social concept (ASC) detection, and that many recent works seem to discuss similar non-concrete entities by using different terminology, in this survey we provide a systematic review of CV work that explicitly or implicitly approaches the problem of abstract (specifically social) concept detection from still images. Specifically, this survey performs and provides: (1) A study and clustering of high level visual understanding semantic elements from a multidisciplinary perspective (computer science, visual studies, and cognitive perspectives); (2) A study and clustering of high level visual understanding computer vision tasks dealing with the identified semantic elements, so as to identify current CV work that implicitly deals with AC detection.

{{</citation>}}


### (34/125) Spatial Transform Decoupling for Oriented Object Detection (Hongtian Yu et al., 2023)

{{<citation>}}

Hongtian Yu, Yunjie Tian, Qixiang Ye, Yunfan Liu. (2023)  
**Spatial Transform Decoupling for Oriented Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10561v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have achieved remarkable success in computer vision tasks. However, their potential in rotation-sensitive scenarios has not been fully explored, and this limitation may be inherently attributed to the lack of spatial invariance in the data-forwarding process. In this study, we present a novel approach, termed Spatial Transform Decoupling (STD), providing a simple-yet-effective solution for oriented object detection with ViTs. Built upon stacked ViT blocks, STD utilizes separate network branches to predict the position, size, and angle of bounding boxes, effectively harnessing the spatial transform potential of ViTs in a divide-and-conquer fashion. Moreover, by aggregating cascaded activation masks (CAMs) computed upon the regressed parameters, STD gradually enhances features within regions of interest (RoIs), which complements the self-attention mechanism. Without bells and whistles, STD achieves state-of-the-art performance on the benchmark datasets including DOTA-v1.0 (82.24% mAP) and HRSC2016 (98.55% mAP), which demonstrates the effectiveness of the proposed method. Source code is available at https://github.com/yuhongtian17/Spatial-Transform-Decoupling.

{{</citation>}}


### (35/125) Improving Diversity in Zero-Shot GAN Adaptation with Semantic Variations (Seogkyu Jeon et al., 2023)

{{<citation>}}

Seogkyu Jeon, Bei Liu, Pilhyeon Lee, Kibeom Hong, Jianlong Fu, Hyeran Byun. (2023)  
**Improving Diversity in Zero-Shot GAN Adaptation with Semantic Variations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.10554v1)  

---


**ABSTRACT**  
Training deep generative models usually requires a large amount of data. To alleviate the data collection cost, the task of zero-shot GAN adaptation aims to reuse well-trained generators to synthesize images of an unseen target domain without any further training samples. Due to the data absence, the textual description of the target domain and the vision-language models, e.g., CLIP, are utilized to effectively guide the generator. However, with only a single representative text feature instead of real images, the synthesized images gradually lose diversity as the model is optimized, which is also known as mode collapse. To tackle the problem, we propose a novel method to find semantic variations of the target text in the CLIP space. Specifically, we explore diverse semantic variations based on the informative text feature of the target domain while regularizing the uncontrolled deviation of the semantic information. With the obtained variations, we design a novel directional moment loss that matches the first and second moments of image and text direction distributions. Moreover, we introduce elastic weight consolidation and a relation consistency loss to effectively preserve valuable content information from the source domain, e.g., appearances. Through extensive experiments, we demonstrate the efficacy of the proposed methods in ensuring sample diversity in various scenarios of zero-shot GAN adaptation. We also conduct ablation studies to validate the effect of each proposed component. Notably, our model achieves a new state-of-the-art on zero-shot GAN adaptation in terms of both diversity and quality.

{{</citation>}}


### (36/125) Joint learning of images and videos with a single Vision Transformer (Shuki Shimizu et al., 2023)

{{<citation>}}

Shuki Shimizu, Toru Tamaki. (2023)  
**Joint learning of images and videos with a single Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10533v1)  

---


**ABSTRACT**  
In this study, we propose a method for jointly learning of images and videos using a single model. In general, images and videos are often trained by separate models. We propose in this paper a method that takes a batch of images as input to Vision Transformer IV-ViT, and also a set of video frames with temporal aggregation by late fusion. Experimental results on two image datasets and two action recognition datasets are presented.

{{</citation>}}


### (37/125) SRFormer: Empowering Regression-Based Text Detection Transformer with Segmentation (Qingwen Bu et al., 2023)

{{<citation>}}

Qingwen Bu, Sungrae Park, Minsoo Khang, Yichuan Cheng. (2023)  
**SRFormer: Empowering Regression-Based Text Detection Transformer with Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10531v1)  

---


**ABSTRACT**  
Existing techniques for text detection can be broadly classified into two primary groups: segmentation-based methods and regression-based methods. Segmentation models offer enhanced robustness to font variations but require intricate post-processing, leading to high computational overhead. Regression-based methods undertake instance-aware prediction but face limitations in robustness and data efficiency due to their reliance on high-level representations. In our academic pursuit, we propose SRFormer, a unified DETR-based model with amalgamated Segmentation and Regression, aiming at the synergistic harnessing of the inherent robustness in segmentation representations, along with the straightforward post-processing of instance-level regression. Our empirical analysis indicates that favorable segmentation predictions can be obtained at the initial decoder layers. In light of this, we constrain the incorporation of segmentation branches to the first few decoder layers and employ progressive regression refinement in subsequent layers, achieving performance gains while minimizing additional computational load from the mask. Furthermore, we propose a Mask-informed Query Enhancement module. We take the segmentation result as a natural soft-ROI to pool and extract robust pixel representations, which are then employed to enhance and diversify instance queries. Extensive experimentation across multiple benchmarks has yielded compelling findings, highlighting our method's exceptional robustness, superior training and data efficiency, as well as its state-of-the-art performance.

{{</citation>}}


### (38/125) Dataset Quantization (Daquan Zhou et al., 2023)

{{<citation>}}

Daquan Zhou, Kai Wang, Jianyang Gu, Xiangyu Peng, Dongze Lian, Yifan Zhang, Yang You, Jiashi Feng. (2023)  
**Dataset Quantization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Quantization  
[Paper Link](http://arxiv.org/abs/2308.10524v1)  

---


**ABSTRACT**  
State-of-the-art deep neural networks are trained with large amounts (millions or even billions) of data. The expensive computation and memory costs make it difficult to train them on limited hardware resources, especially for recent popular large language models (LLM) and computer vision models (CV). Recent popular dataset distillation methods are thus developed, aiming to reduce the number of training samples via synthesizing small-scale datasets via gradient matching. However, as the gradient calculation is coupled with the specific network architecture, the synthesized dataset is biased and performs poorly when used for training unseen architectures. To address these limitations, we present dataset quantization (DQ), a new framework to compress large-scale datasets into small subsets which can be used for training any neural network architectures. Extensive experiments demonstrate that DQ is able to generate condensed small datasets for training unseen network architectures with state-of-the-art compression ratios for lossless model training. To the best of our knowledge, DQ is the first method that can successfully distill large-scale datasets such as ImageNet-1k with a state-of-the-art compression ratio. Notably, with 60% data from ImageNet and 20% data from Alpaca's instruction tuning data, the models can be trained with negligible or no performance drop for both vision tasks (including classification, semantic segmentation, and object detection) as well as language tasks (including instruction tuning tasks such as BBH and DROP).

{{</citation>}}


### (39/125) PHE-SICH-CT-IDS: A Benchmark CT Image Dataset for Evaluation Semantic Segmentation, Object Detection and Radiomic Feature Extraction of Perihematomal Edema in Spontaneous Intracerebral Hemorrhage (Deguo Ma et al., 2023)

{{<citation>}}

Deguo Ma, Chen Li, Lin Qiao, Tianming Du, Dechao Tang, Zhiyu Ma, Marcin Grzegorzek Hongzan, Hongzan Sun. (2023)  
**PHE-SICH-CT-IDS: A Benchmark CT Image Dataset for Evaluation Semantic Segmentation, Object Detection and Radiomic Feature Extraction of Perihematomal Edema in Spontaneous Intracerebral Hemorrhage**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.10521v1)  

---


**ABSTRACT**  
Intracerebral hemorrhage is one of the diseases with the highest mortality and poorest prognosis worldwide. Spontaneous intracerebral hemorrhage (SICH) typically presents acutely, prompt and expedited radiological examination is crucial for diagnosis, localization, and quantification of the hemorrhage. Early detection and accurate segmentation of perihematomal edema (PHE) play a critical role in guiding appropriate clinical intervention and enhancing patient prognosis. However, the progress and assessment of computer-aided diagnostic methods for PHE segmentation and detection face challenges due to the scarcity of publicly accessible brain CT image datasets. This study establishes a publicly available CT dataset named PHE-SICH-CT-IDS for perihematomal edema in spontaneous intracerebral hemorrhage. The dataset comprises 120 brain CT scans and 7,022 CT images, along with corresponding medical information of the patients. To demonstrate its effectiveness, classical algorithms for semantic segmentation, object detection, and radiomic feature extraction are evaluated. The experimental results confirm the suitability of PHE-SICH-CT-IDS for assessing the performance of segmentation, detection and radiomic feature extraction methods. To the best of our knowledge, this is the first publicly available dataset for PHE in SICH, comprising various data formats suitable for applications across diverse medical scenarios. We believe that PHE-SICH-CT-IDS will allure researchers to explore novel algorithms, providing valuable support for clinicians and patients in the clinical setting. PHE-SICH-CT-IDS is freely published for non-commercial purpose at: https://figshare.com/articles/dataset/PHE-SICH-CT-IDS/23957937.

{{</citation>}}


### (40/125) QD-BEV : Quantization-aware View-guided Distillation for Multi-view 3D Object Detection (Yifan Zhang et al., 2023)

{{<citation>}}

Yifan Zhang, Zhen Dong, Huanrui Yang, Ming Lu, Cheng-Ching Tseng, Yuan Du, Kurt Keutzer, Li Du, Shanghang Zhang. (2023)  
**QD-BEV : Quantization-aware View-guided Distillation for Multi-view 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2308.10515v1)  

---


**ABSTRACT**  
Multi-view 3D detection based on BEV (bird-eye-view) has recently achieved significant improvements. However, the huge memory consumption of state-of-the-art models makes it hard to deploy them on vehicles, and the non-trivial latency will affect the real-time perception of streaming applications. Despite the wide application of quantization to lighten models, we show in our paper that directly applying quantization in BEV tasks will 1) make the training unstable, and 2) lead to intolerable performance degradation. To solve these issues, our method QD-BEV enables a novel view-guided distillation (VGD) objective, which can stabilize the quantization-aware training (QAT) while enhancing the model performance by leveraging both image features and BEV features. Our experiments show that QD-BEV achieves similar or even better accuracy than previous methods with significant efficiency gains. On the nuScenes datasets, the 4-bit weight and 6-bit activation quantized QD-BEV-Tiny model achieves 37.2% NDS with only 15.8 MB model size, outperforming BevFormer-Tiny by 1.8% with an 8x model compression. On the Small and Base variants, QD-BEV models also perform superbly and achieve 47.9% NDS (28.2 MB) and 50.9% NDS (32.9 MB), respectively.

{{</citation>}}


### (41/125) Semantic Graph Representation Learning for Handwritten Mathematical Expression Recognition (Zhuang Liu et al., 2023)

{{<citation>}}

Zhuang Liu, Ye Yuan, Zhilong Ji, Jingfeng Bai, Xiang Bai. (2023)  
**Semantic Graph Representation Learning for Handwritten Mathematical Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.10493v1)  

---


**ABSTRACT**  
Handwritten mathematical expression recognition (HMER) has attracted extensive attention recently. However, current methods cannot explicitly study the interactions between different symbols, which may fail when faced similar symbols. To alleviate this issue, we propose a simple but efficient method to enhance semantic interaction learning (SIL). Specifically, we firstly construct a semantic graph based on the statistical symbol co-occurrence probabilities. Then we design a semantic aware module (SAM), which projects the visual and classification feature into semantic space. The cosine distance between different projected vectors indicates the correlation between symbols. And jointly optimizing HMER and SIL can explicitly enhances the model's understanding of symbol relationships. In addition, SAM can be easily plugged into existing attention-based models for HMER and consistently bring improvement. Extensive experiments on public benchmark datasets demonstrate that our proposed module can effectively enhance the recognition performance. Our method achieves better recognition performance than prior arts on both CROHME and HME100K datasets.

{{</citation>}}


### (42/125) SynDrone -- Multi-modal UAV Dataset for Urban Scenarios (Giulia Rizzoli et al., 2023)

{{<citation>}}

Giulia Rizzoli, Francesco Barbato, Matteo Caligiuri, Pietro Zanuttigh. (2023)  
**SynDrone -- Multi-modal UAV Dataset for Urban Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.10491v1)  

---


**ABSTRACT**  
The development of computer vision algorithms for Unmanned Aerial Vehicles (UAVs) imagery heavily relies on the availability of annotated high-resolution aerial data. However, the scarcity of large-scale real datasets with pixel-level annotations poses a significant challenge to researchers as the limited number of images in existing datasets hinders the effectiveness of deep learning models that require a large amount of training data. In this paper, we propose a multimodal synthetic dataset containing both images and 3D data taken at multiple flying heights to address these limitations. In addition to object-level annotations, the provided data also include pixel-level labeling in 28 classes, enabling exploration of the potential advantages in tasks like semantic segmentation. In total, our dataset contains 72k labeled samples that allow for effective training of deep architectures showing promising results in synthetic-to-real adaptation. The dataset will be made publicly available to support the development of novel computer vision methods targeting UAV applications.

{{</citation>}}


### (43/125) ADNet: Lane Shape Prediction via Anchor Decomposition (Lingyu Xiao et al., 2023)

{{<citation>}}

Lingyu Xiao, Xiang Li, Sen Yang, Wankou Yang. (2023)  
**ADNet: Lane Shape Prediction via Anchor Decomposition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.10481v1)  

---


**ABSTRACT**  
In this paper, we revisit the limitations of anchor-based lane detection methods, which have predominantly focused on fixed anchors that stem from the edges of the image, disregarding their versatility and quality. To overcome the inflexibility of anchors, we decompose them into learning the heat map of starting points and their associated directions. This decomposition removes the limitations on the starting point of anchors, making our algorithm adaptable to different lane types in various datasets. To enhance the quality of anchors, we introduce the Large Kernel Attention (LKA) for Feature Pyramid Network (FPN). This significantly increases the receptive field, which is crucial in capturing the sufficient context as lane lines typically run throughout the entire image. We have named our proposed system the Anchor Decomposition Network (ADNet). Additionally, we propose the General Lane IoU (GLIoU) loss, which significantly improves the performance of ADNet in complex scenarios. Experimental results on three widely used lane detection benchmarks, VIL-100, CULane, and TuSimple, demonstrate that our approach outperforms the state-of-the-art methods on VIL-100 and exhibits competitive accuracy on CULane and TuSimple. Code and models will be released on https://github.com/ Sephirex-X/ADNet.

{{</citation>}}


### (44/125) CVFC: Attention-Based Cross-View Feature Consistency for Weakly Supervised Semantic Segmentation of Pathology Images (Liangrui Pan et al., 2023)

{{<citation>}}

Liangrui Pan, Lian Wang, Zhichao Feng, Liwen Xu, Shaoliang Peng. (2023)  
**CVFC: Attention-Based Cross-View Feature Consistency for Weakly Supervised Semantic Segmentation of Pathology Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.10449v1)  

---


**ABSTRACT**  
Histopathology image segmentation is the gold standard for diagnosing cancer, and can indicate cancer prognosis. However, histopathology image segmentation requires high-quality masks, so many studies now use imagelevel labels to achieve pixel-level segmentation to reduce the need for fine-grained annotation. To solve this problem, we propose an attention-based cross-view feature consistency end-to-end pseudo-mask generation framework named CVFC based on the attention mechanism. Specifically, CVFC is a three-branch joint framework composed of two Resnet38 and one Resnet50, and the independent branch multi-scale integrated feature map to generate a class activation map (CAM); in each branch, through down-sampling and The expansion method adjusts the size of the CAM; the middle branch projects the feature matrix to the query and key feature spaces, and generates a feature space perception matrix through the connection layer and inner product to adjust and refine the CAM of each branch; finally, through the feature consistency loss and feature cross loss to optimize the parameters of CVFC in co-training mode. After a large number of experiments, An IoU of 0.7122 and a fwIoU of 0.7018 are obtained on the WSSS4LUAD dataset, which outperforms HistoSegNet, SEAM, C-CAM, WSSS-Tissue, and OEEM, respectively.

{{</citation>}}


### (45/125) When Prompt-based Incremental Learning Does Not Meet Strong Pretraining (Yu-Ming Tang et al., 2023)

{{<citation>}}

Yu-Ming Tang, Yi-Xing Peng, Wei-Shi Zheng. (2023)  
**When Prompt-based Incremental Learning Does Not Meet Strong Pretraining**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.10445v1)  

---


**ABSTRACT**  
Incremental learning aims to overcome catastrophic forgetting when learning deep networks from sequential tasks. With impressive learning efficiency and performance, prompt-based methods adopt a fixed backbone to sequential tasks by learning task-specific prompts. However, existing prompt-based methods heavily rely on strong pretraining (typically trained on ImageNet-21k), and we find that their models could be trapped if the potential gap between the pretraining task and unknown future tasks is large. In this work, we develop a learnable Adaptive Prompt Generator (APG). The key is to unify the prompt retrieval and prompt learning processes into a learnable prompt generator. Hence, the whole prompting process can be optimized to reduce the negative effects of the gap between tasks effectively. To make our APG avoid learning ineffective knowledge, we maintain a knowledge pool to regularize APG with the feature distribution of each class. Extensive experiments show that our method significantly outperforms advanced methods in exemplar-free incremental learning without (strong) pretraining. Besides, under strong retraining, our method also has comparable performance to existing prompt-based models, showing that our method can still benefit from pretraining. Codes can be found at https://github.com/TOM-tym/APG

{{</citation>}}


### (46/125) Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks (Kaixin Xu et al., 2023)

{{<citation>}}

Kaixin Xu, Zhe Wang, Xue Geng, Jie Lin, Min Wu, Xiaoli Li, Weisi Lin. (2023)  
**Efficient Joint Optimization of Layer-Adaptive Weight Pruning in Deep Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2308.10438v2)  

---


**ABSTRACT**  
In this paper, we propose a novel layer-adaptive weight-pruning approach for Deep Neural Networks (DNNs) that addresses the challenge of optimizing the output distortion minimization while adhering to a target pruning ratio constraint. Our approach takes into account the collective influence of all layers to design a layer-adaptive pruning scheme. We discover and utilize a very important additivity property of output distortion caused by pruning weights on multiple layers. This property enables us to formulate the pruning as a combinatorial optimization problem and efficiently solve it through dynamic programming. By decomposing the problem into sub-problems, we achieve linear time complexity, making our optimization algorithm fast and feasible to run on CPUs. Our extensive experiments demonstrate the superiority of our approach over existing methods on the ImageNet and CIFAR-10 datasets. On CIFAR-10, our method achieves remarkable improvements, outperforming others by up to 1.0% for ResNet-32, 0.5% for VGG-16, and 0.7% for DenseNet-121 in terms of top-1 accuracy. On ImageNet, we achieve up to 4.7% and 4.6% higher top-1 accuracy compared to other methods for VGG-16 and ResNet-50, respectively. These results highlight the effectiveness and practicality of our approach for enhancing DNN performance through layer-adaptive weight pruning. Code will be available on https://github.com/Akimoto-Cris/RD_VIT_PRUNE.

{{</citation>}}


### (47/125) Simple Baselines for Interactive Video Retrieval with Questions and Answers (Kaiqu Liang et al., 2023)

{{<citation>}}

Kaiqu Liang, Samuel Albanie. (2023)  
**Simple Baselines for Interactive Video Retrieval with Questions and Answers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-HC, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.10402v1)  

---


**ABSTRACT**  
To date, the majority of video retrieval systems have been optimized for a "single-shot" scenario in which the user submits a query in isolation, ignoring previous interactions with the system. Recently, there has been renewed interest in interactive systems to enhance retrieval, but existing approaches are complex and deliver limited gains in performance. In this work, we revisit this topic and propose several simple yet effective baselines for interactive video retrieval via question-answering. We employ a VideoQA model to simulate user interactions and show that this enables the productive study of the interactive retrieval task without access to ground truth dialogue data. Experiments on MSR-VTT, MSVD, and AVSD show that our framework using question-based interaction significantly improves the performance of text-based video retrieval systems.

{{</citation>}}


## cs.AI (8)



### (48/125) Neural Amortized Inference for Nested Multi-agent Reasoning (Kunal Jha et al., 2023)

{{<citation>}}

Kunal Jha, Tuan Anh Le, Chuanyang Jin, Yen-Ling Kuo, Joshua B. Tenenbaum, Tianmin Shu. (2023)  
**Neural Amortized Inference for Nested Multi-agent Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs-RO, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.11071v1)  

---


**ABSTRACT**  
Multi-agent interactions, such as communication, teaching, and bluffing, often rely on higher-order social inference, i.e., understanding how others infer oneself. Such intricate reasoning can be effectively modeled through nested multi-agent reasoning. Nonetheless, the computational complexity escalates exponentially with each level of reasoning, posing a significant challenge. However, humans effortlessly perform complex social inferences as part of their daily lives. To bridge the gap between human-like inference capabilities and computational limitations, we propose a novel approach: leveraging neural networks to amortize high-order social inference, thereby expediting nested multi-agent reasoning. We evaluate our method in two challenging multi-agent interaction domains. The experimental results demonstrate that our method is computationally efficient while exhibiting minimal degradation in accuracy.

{{</citation>}}


### (49/125) CSM-H-R: An Automatic Context Reasoning Framework for Interoperable Intelligent Systems and Privacy Protection (Songhui Yue et al., 2023)

{{<citation>}}

Songhui Yue, Xiaoyan Hong, Randy K. Smith. (2023)  
**CSM-H-R: An Automatic Context Reasoning Framework for Interoperable Intelligent Systems and Privacy Protection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SY, cs.AI, eess-SY  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.11066v1)  

---


**ABSTRACT**  
Automation of High-Level Context (HLC) reasoning for intelligent systems at scale is imperative due to the unceasing accumulation of contextual data in the IoT era, the trend of the fusion of data from multi-sources, and the intrinsic complexity and dynamism of the context-based decision-making process. To mitigate this issue, we propose an automatic context reasoning framework CSM-H-R, which programmatically combines ontologies and states at runtime and the model-storage phase for attaining the ability to recognize meaningful HLC, and the resulting data representation can be applied to different reasoning techniques. Case studies are developed based on an intelligent elevator system in a smart campus setting. An implementation of the framework - a CSM Engine, and the experiments of translating the HLC reasoning into vector and matrix computing especially take care of the dynamic aspects of context and present the potentiality of using advanced mathematical and probabilistic models to achieve the next level of automation in integrating intelligent systems; meanwhile, privacy protection support is achieved by anonymization through label embedding and reducing information correlation. The code of this study is available at: https://github.com/songhui01/CSM-H-R.

{{</citation>}}


### (50/125) 'Guinea Pig Trials' Utilizing GPT: A Novel Smart Agent-Based Modeling Approach for Studying Firm Competition and Collusion (Xu Han et al., 2023)

{{<citation>}}

Xu Han, Zengqing Wu, Chuan Xiao. (2023)  
**'Guinea Pig Trials' Utilizing GPT: A Novel Smart Agent-Based Modeling Approach for Studying Firm Competition and Collusion**  

---
Primary Category: cs.AI  
Categories: I-2-1; I-2-7; I-6-5; J-4, cs-AI, cs-CE, cs-MA, cs.AI, econ-GN, q-fin-EC  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.10974v1)  

---


**ABSTRACT**  
Firm competition and collusion involve complex dynamics, particularly when considering communication among firms. Such issues can be modeled as problems of complex systems, traditionally approached through experiments involving human subjects or agent-based modeling methods. We propose an innovative framework called Smart Agent-Based Modeling (SABM), wherein smart agents, supported by GPT-4 technologies, represent firms, and interact with one another. We conducted a controlled experiment to study firm price competition and collusion behaviors under various conditions. SABM is more cost-effective and flexible compared to conducting experiments with human subjects. Smart agents possess an extensive knowledge base for decision-making and exhibit human-like strategic abilities, surpassing traditional ABM agents. Furthermore, smart agents can simulate human conversation and be personalized, making them ideal for studying complex situations involving communication. Our results demonstrate that, in the absence of communication, smart agents consistently reach tacit collusion, leading to prices converging at levels higher than the Bertrand equilibrium price but lower than monopoly or cartel prices. When communication is allowed, smart agents achieve a higher-level collusion with prices close to cartel prices. Collusion forms more quickly with communication, while price convergence is smoother without it. These results indicate that communication enhances trust between firms, encouraging frequent small price deviations to explore opportunities for a higher-level win-win situation and reducing the likelihood of triggering a price war. We also assigned different personas to firms to analyze behavioral differences and tested variant models under diverse market structures. The findings showcase the effectiveness and robustness of SABM and provide intriguing insights into competition and collusion.

{{</citation>}}


### (51/125) Giraffe: Adventures in Expanding Context Lengths in LLMs (Arka Pal et al., 2023)

{{<citation>}}

Arka Pal, Deep Karkhanis, Manley Roberts, Samuel Dooley, Arvind Sundararajan, Siddartha Naidu. (2023)  
**Giraffe: Adventures in Expanding Context Lengths in LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: LLaMA, QA  
[Paper Link](http://arxiv.org/abs/2308.10882v1)  

---


**ABSTRACT**  
Modern large language models (LLMs) that rely on attention mechanisms are typically trained with fixed context lengths which enforce upper limits on the length of input sequences that they can handle at evaluation time. To use these models on sequences longer than the train-time context length, one might employ techniques from the growing family of context length extrapolation methods -- most of which focus on modifying the system of positional encodings used in the attention mechanism to indicate where tokens or activations are located in the input sequence. We conduct a wide survey of existing methods of context length extrapolation on a base LLaMA or LLaMA 2 model, and introduce some of our own design as well -- in particular, a new truncation strategy for modifying the basis for the position encoding.   We test these methods using three new evaluation tasks (FreeFormQA, AlteredNumericQA, and LongChat-Lines) as well as perplexity, which we find to be less fine-grained as a measure of long context performance of LLMs. We release the three tasks publicly as datasets on HuggingFace. We discover that linear scaling is the best method for extending context length, and show that further gains can be achieved by using longer scales at evaluation time. We also discover promising extrapolation capabilities in the truncated basis. To support further research in this area, we release three new 13B parameter long-context models which we call Giraffe: 4k and 16k context models trained from base LLaMA-13B, and a 32k context model trained from base LLaMA2-13B. We also release the code to replicate our results.

{{</citation>}}


### (52/125) KGrEaT: A Framework to Evaluate Knowledge Graphs via Downstream Tasks (Nicolas Heist et al., 2023)

{{<citation>}}

Nicolas Heist, Sven Hertling, Heiko Paulheim. (2023)  
**KGrEaT: A Framework to Evaluate Knowledge Graphs via Downstream Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DB, cs-LG, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.10537v1)  

---


**ABSTRACT**  
In recent years, countless research papers have addressed the topics of knowledge graph creation, extension, or completion in order to create knowledge graphs that are larger, more correct, or more diverse. This research is typically motivated by the argumentation that using such enhanced knowledge graphs to solve downstream tasks will improve performance. Nonetheless, this is hardly ever evaluated. Instead, the predominant evaluation metrics - aiming at correctness and completeness - are undoubtedly valuable but fail to capture the complete picture, i.e., how useful the created or enhanced knowledge graph actually is. Further, the accessibility of such a knowledge graph is rarely considered (e.g., whether it contains expressive labels, descriptions, and sufficient context information to link textual mentions to the entities of the knowledge graph). To better judge how well knowledge graphs perform on actual tasks, we present KGrEaT - a framework to estimate the quality of knowledge graphs via actual downstream tasks like classification, clustering, or recommendation. Instead of comparing different methods of processing knowledge graphs with respect to a single task, the purpose of KGrEaT is to compare various knowledge graphs as such by evaluating them on a fixed task setup. The framework takes a knowledge graph as input, automatically maps it to the datasets to be evaluated on, and computes performance metrics for the defined tasks. It is built in a modular way to be easily extendable with additional tasks and datasets.

{{</citation>}}


### (53/125) Elucidating STEM Concepts through Generative AI: A Multi-modal Exploration of Analogical Reasoning (Chen Cao et al., 2023)

{{<citation>}}

Chen Cao, Zijian Ding, Gyeong-Geon Lee, Jiajun Jiao, Jionghao Lin, Xiaoming Zhai. (2023)  
**Elucidating STEM Concepts through Generative AI: A Multi-modal Exploration of Analogical Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-HC, cs.AI  
Keywords: AI, Generative AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.10454v1)  

---


**ABSTRACT**  
This study explores the integration of generative artificial intelligence (AI), specifically large language models, with multi-modal analogical reasoning as an innovative approach to enhance science, technology, engineering, and mathematics (STEM) education. We have developed a novel system that utilizes the capacities of generative AI to transform intricate principles in mathematics, physics, and programming into comprehensible metaphors. To further augment the educational experience, these metaphors are subsequently converted into visual form. Our study aims to enhance the learners' understanding of STEM concepts and their learning engagement by using the visual metaphors. We examine the efficacy of our system via a randomized A/B/C test, assessing learning gains and motivation shifts among the learners. Our study demonstrates the potential of applying large language models to educational practice on STEM subjects. The results will shed light on the design of educational system in terms of harnessing AI's potential to empower educational stakeholders.

{{</citation>}}


### (54/125) Using Large Language Models for Cybersecurity Capture-The-Flag Challenges and Certification Questions (Wesley Tann et al., 2023)

{{<citation>}}

Wesley Tann, Yuancheng Liu, Jun Heng Sim, Choon Meng Seah, Ee-Chien Chang. (2023)  
**Using Large Language Models for Cybersecurity Capture-The-Flag Challenges and Certification Questions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs.AI  
Keywords: AI, ChatGPT, GPT, Google, Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2308.10443v1)  

---


**ABSTRACT**  
The assessment of cybersecurity Capture-The-Flag (CTF) exercises involves participants finding text strings or ``flags'' by exploiting system vulnerabilities. Large Language Models (LLMs) are natural-language models trained on vast amounts of words to understand and generate text; they can perform well on many CTF challenges. Such LLMs are freely available to students. In the context of CTF exercises in the classroom, this raises concerns about academic integrity. Educators must understand LLMs' capabilities to modify their teaching to accommodate generative AI assistance. This research investigates the effectiveness of LLMs, particularly in the realm of CTF challenges and questions. Here we evaluate three popular LLMs, OpenAI ChatGPT, Google Bard, and Microsoft Bing. First, we assess the LLMs' question-answering performance on five Cisco certifications with varying difficulty levels. Next, we qualitatively study the LLMs' abilities in solving CTF challenges to understand their limitations. We report on the experience of using the LLMs for seven test cases in all five types of CTF challenges. In addition, we demonstrate how jailbreak prompts can bypass and break LLMs' ethical safeguards. The paper concludes by discussing LLM's impact on CTF exercises and its implications.

{{</citation>}}


### (55/125) X-VoE: Measuring eXplanatory Violation of Expectation in Physical Events (Bo Dai et al., 2023)

{{<citation>}}

Bo Dai, Linge Wang, Baoxiong Jia, Zeyu Zhang, Song-Chun Zhu, Chi Zhang, Yixin Zhu. (2023)  
**X-VoE: Measuring eXplanatory Violation of Expectation in Physical Events**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10441v1)  

---


**ABSTRACT**  
Intuitive physics is pivotal for human understanding of the physical world, enabling prediction and interpretation of events even in infancy. Nonetheless, replicating this level of intuitive physics in artificial intelligence (AI) remains a formidable challenge. This study introduces X-VoE, a comprehensive benchmark dataset, to assess AI agents' grasp of intuitive physics. Built on the developmental psychology-rooted Violation of Expectation (VoE) paradigm, X-VoE establishes a higher bar for the explanatory capacities of intuitive physics models. Each VoE scenario within X-VoE encompasses three distinct settings, probing models' comprehension of events and their underlying explanations. Beyond model evaluation, we present an explanation-based learning system that captures physics dynamics and infers occluded object states solely from visual sequences, without explicit occlusion labels. Experimental outcomes highlight our model's alignment with human commonsense when tested against X-VoE. A remarkable feature is our model's ability to visually expound VoE events by reconstructing concealed scenes. Concluding, we discuss the findings' implications and outline future research directions. Through X-VoE, we catalyze the advancement of AI endowed with human-like intuitive physics capabilities.

{{</citation>}}


## cs.LG (24)



### (56/125) Topological Graph Signal Compression (Guillermo Bernárdez et al., 2023)

{{<citation>}}

Guillermo Bernárdez, Lev Telyatnikov, Eduard Alarcón, Albert Cabellos-Aparicio, Pere Barlet-Ros, Pietro Liò. (2023)  
**Topological Graph Signal Compression**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.11068v1)  

---


**ABSTRACT**  
Recently emerged Topological Deep Learning (TDL) methods aim to extend current Graph Neural Networks (GNN) by naturally processing higher-order interactions, going beyond the pairwise relations and local neighborhoods defined by graph representations. In this paper we propose a novel TDL-based method for compressing signals over graphs, consisting in two main steps: first, disjoint sets of higher-order structures are inferred based on the original signal --by clustering $N$ datapoints into $K\ll N$ collections; then, a topological-inspired message passing gets a compressed representation of the signal within those multi-element sets. Our results show that our framework improves both standard GNN and feed-forward architectures in compressing temporal link-based signals from two real-word Internet Service Provider Networks' datasets --from $30\%$ up to $90\%$ better reconstruction errors across all evaluation scenarios--, suggesting that it better captures and exploits spatial and temporal correlations over the whole graph-based network structure.

{{</citation>}}


### (57/125) FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning (Haokun Chen et al., 2023)

{{<citation>}}

Haokun Chen, Yao Zhang, Denis Krompass, Jindong Gu, Volker Tresp. (2023)  
**FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MM, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.12305v1)  

---


**ABSTRACT**  
Recently, foundation models have exhibited remarkable advancements in multi-modal learning. These models, equipped with millions (or billions) of parameters, typically require a substantial amount of data for finetuning. However, collecting and centralizing training data from diverse sectors becomes challenging due to distinct privacy regulations. Federated Learning (FL) emerges as a promising solution, enabling multiple clients to collaboratively train neural networks without centralizing their local data. To alleviate client computation burdens and communication overheads, previous works have adapted Parameter-efficient Finetuning (PEFT) methods for FL. Hereby, only a small fraction of the model parameters are optimized and communicated during federated communications. Nevertheless, most previous works have focused on a single modality and neglected one common phenomenon, i.e., the presence of data heterogeneity across the clients. Therefore, in this work, we propose a finetuning framework tailored to heterogeneous multi-modal FL, called Federated Dual-Aadapter Teacher (FedDAT). Specifically, our approach leverages a Dual-Adapter Teacher (DAT) to address data heterogeneity by regularizing the client local updates and applying Mutual Knowledge Distillation (MKD) for an efficient knowledge transfer. FedDAT is the first approach that enables an efficient distributed finetuning of foundation models for a variety of heterogeneous Vision-Language tasks. To demonstrate its effectiveness, we conduct extensive experiments on four multi-modality FL benchmarks with different types of data heterogeneity, where FedDAT substantially outperforms the existing centralized PEFT methods adapted for FL.

{{</citation>}}


### (58/125) Personalized Event Prediction for Electronic Health Records (Jeong Min Lee et al., 2023)

{{<citation>}}

Jeong Min Lee, Milos Hauskrecht. (2023)  
**Personalized Event Prediction for Electronic Health Records**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-AP  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.11013v1)  

---


**ABSTRACT**  
Clinical event sequences consist of hundreds of clinical events that represent records of patient care in time. Developing accurate predictive models of such sequences is of a great importance for supporting a variety of models for interpreting/classifying the current patient condition, or predicting adverse clinical events and outcomes, all aimed to improve patient care. One important challenge of learning predictive models of clinical sequences is their patient-specific variability. Based on underlying clinical conditions, each patient's sequence may consist of different sets of clinical events (observations, lab results, medications, procedures). Hence, simple population-wide models learned from event sequences for many different patients may not accurately predict patient-specific dynamics of event sequences and their differences. To address the problem, we propose and investigate multiple new event sequence prediction models and methods that let us better adjust the prediction for individual patients and their specific conditions. The methods developed in this work pursue refinement of population-wide models to subpopulations, self-adaptation, and a meta-level model switching that is able to adaptively select the model with the best chance to support the immediate prediction. We analyze and test the performance of these models on clinical event sequences of patients in MIMIC-III database.

{{</citation>}}


### (59/125) Unlocking Accuracy and Fairness in Differentially Private Image Classification (Leonard Berrada et al., 2023)

{{<citation>}}

Leonard Berrada, Soham De, Judy Hanwen Shen, Jamie Hayes, Robert Stanforth, David Stutz, Pushmeet Kohli, Samuel L. Smith, Borja Balle. (2023)  
**Unlocking Accuracy and Fairness in Differentially Private Image Classification**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-CY, cs-LG, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2308.10888v1)  

---


**ABSTRACT**  
Privacy-preserving machine learning aims to train models on private data without leaking sensitive information. Differential privacy (DP) is considered the gold standard framework for privacy-preserving training, as it provides formal privacy guarantees. However, compared to their non-private counterparts, models trained with DP often have significantly reduced accuracy. Private classifiers are also believed to exhibit larger performance disparities across subpopulations, raising fairness concerns. The poor performance of classifiers trained with DP has prevented the widespread adoption of privacy preserving machine learning in industry. Here we show that pre-trained foundation models fine-tuned with DP can achieve similar accuracy to non-private classifiers, even in the presence of significant distribution shifts between pre-training data and downstream tasks. We achieve private accuracies within a few percent of the non-private state of the art across four datasets, including two medical imaging benchmarks. Furthermore, our private medical classifiers do not exhibit larger performance disparities across demographic groups than non-private models. This milestone to make DP training a practical and reliable technology has the potential to widely enable machine learning practitioners to train safely on sensitive datasets while protecting individuals' privacy.

{{</citation>}}


### (60/125) Analyzing Transformer Dynamics as Movement through Embedding Space (Sumeet S. Singh, 2023)

{{<citation>}}

Sumeet S. Singh. (2023)  
**Analyzing Transformer Dynamics as Movement through Embedding Space**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs.LG  
Keywords: Attention, Embedding, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10874v1)  

---


**ABSTRACT**  
Transformer language models exhibit intelligent behaviors such as understanding natural language, recognizing patterns, acquiring knowledge, reasoning, planning, reflecting and using tools. This paper explores how their underlying mechanics give rise to intelligent behaviors. We adopt a systems approach to analyze Transformers in detail and develop a mathematical framework that frames their dynamics as movement through embedding space. This novel perspective provides a principled way of thinking about the problem and reveals important insights related to the emergence of intelligence:   1. At its core the Transformer is a Embedding Space walker, mapping intelligent behavior to trajectories in this vector space.   2. At each step of the walk, it composes context into a single composite vector whose location in Embedding Space defines the next step.   3. No learning actually occurs during decoding; in-context learning and generalization are simply the result of different contexts composing into different vectors.   4. Ultimately the knowledge, intelligence and skills exhibited by the model are embodied in the organization of vectors in Embedding Space rather than in specific neurons or layers. These abilities are properties of this organization.   5. Attention's contribution boils down to the association-bias it lends to vector composition and which influences the aforementioned organization. However, more investigation is needed to ascertain its significance.   6. The entire model is composed from two principal operations: data independent filtering and data dependent aggregation. This generalization unifies Transformers with other sequence models and across modalities.   Building upon this foundation we formalize and test a semantic space theory which posits that embedding vectors represent semantic concepts and find some evidence of its validity.

{{</citation>}}


### (61/125) Majorana Demonstrator Data Release for AI/ML Applications (I. J. Arnquist et al., 2023)

{{<citation>}}

I. J. Arnquist, F. T. Avignone III, A. S. Barabash, C. J. Barton, K. H. Bhimani, E. Blalock, B. Bos, M. Busch, M. Buuck, T. S. Caldwell, Y. -D. Chan, C. D. Christofferson, P. -H. Chu, M. L. Clark, C. Cuesta, J. A. Detwiler, Yu. Efremenko, H. Ejiri, S. R. Elliott, N. Fuad, G. K. Giovanetti, M. P. Green, J. Gruszko, I. S. Guinn, V. E. Guiseppe, C. R. Haufe, R. Henning, D. Hervas Aguilar, E. W. Hoppe, A. Hostiuc, M. F. Kidd, I. Kim, R. T. Kouzes, T. E. Lannen V, A. Li, J. M. Lopez-Castano, R. D. Martin, R. Massarczyk, S. J. Meijer, S. Mertens, T. K. Oli, L. S. Paudel, W. Pettus, A. W. P. Poon, B. Quenallata, D. C. Radford, A. L. Reine, K. Rielage, N. W. Ruof, D. C. Schaper, S. J. Schleich, D. Tedeschi, R. L. Varner, S. Vasilyev, S. L. Watkins, J. F. Wilkerson, C. Wiseman, W. Xu, C. -H. Yu, B. X. Zhu. (2023)  
**Majorana Demonstrator Data Release for AI/ML Applications**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, nucl-ex, physics-data-an, physics-ins-det  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10856v2)  

---


**ABSTRACT**  
The enclosed data release consists of a subset of the calibration data from the Majorana Demonstrator experiment. Each Majorana event is accompanied by raw Germanium detector waveforms, pulse shape discrimination cuts, and calibrated final energies, all shared in an HDF5 file format along with relevant metadata. This release is specifically designed to support the training and testing of Artificial Intelligence (AI) and Machine Learning (ML) algorithms upon our data. This document is structured as follows. Section I provides an overview of the dataset's content and format; Section II outlines the location of this dataset and the method for accessing it; Section III presents the NPML Machine Learning Challenge associated with this dataset; Section IV contains a disclaimer from the Majorana collaboration regarding the use of this dataset; Appendix A contains technical details of this data release. Please direct questions about the material provided within this release to liaobo77@ucsd.edu (A. Li).

{{</citation>}}


### (62/125) Real World Time Series Benchmark Datasets with Distribution Shifts: Global Crude Oil Price and Volatility (Pranay Pasula, 2023)

{{<citation>}}

Pranay Pasula. (2023)  
**Real World Time Series Benchmark Datasets with Distribution Shifts: Global Crude Oil Price and Volatility**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.10846v1)  

---


**ABSTRACT**  
The scarcity of task-labeled time-series benchmarks in the financial domain hinders progress in continual learning. Addressing this deficit would foster innovation in this area. Therefore, we present COB, Crude Oil Benchmark datasets. COB includes 30 years of asset prices that exhibit significant distribution shifts and optimally generates corresponding task (i.e., regime) labels based on these distribution shifts for the three most important crude oils in the world. Our contributions include creating real-world benchmark datasets by transforming asset price data into volatility proxies, fitting models using expectation-maximization (EM), generating contextual task labels that align with real-world events, and providing these labels as well as the general algorithm to the public. We show that the inclusion of these task labels universally improves performance on four continual learning algorithms, some state-of-the-art, over multiple forecasting horizons. We hope these benchmarks accelerate research in handling distribution shifts in real-world data, especially due to the global importance of the assets considered. We've made the (1) raw price data, (2) task labels generated by our approach, (3) and code for our algorithm available at https://oilpricebenchmarks.github.io.

{{</citation>}}


### (63/125) Graph Neural Bandits (Yunzhe Qi et al., 2023)

{{<citation>}}

Yunzhe Qi, Yikun Ban, Jingrui He. (2023)  
**Graph Neural Bandits**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.10808v1)  

---


**ABSTRACT**  
Contextual bandits algorithms aim to choose the optimal arm with the highest reward out of a set of candidates based on the contextual information. Various bandit algorithms have been applied to real-world applications due to their ability of tackling the exploitation-exploration dilemma. Motivated by online recommendation scenarios, in this paper, we propose a framework named Graph Neural Bandits (GNB) to leverage the collaborative nature among users empowered by graph neural networks (GNNs). Instead of estimating rigid user clusters as in existing works, we model the "fine-grained" collaborative effects through estimated user graphs in terms of exploitation and exploration respectively. Then, to refine the recommendation strategy, we utilize separate GNN-based models on estimated user graphs for exploitation and adaptive exploration. Theoretical analysis and experimental results on multiple real data sets in comparison with state-of-the-art baselines are provided to demonstrate the effectiveness of our proposed framework.

{{</citation>}}


### (64/125) Stabilizing Unsupervised Environment Design with a Learned Adversary (Ishita Mediratta et al., 2023)

{{<citation>}}

Ishita Mediratta, Minqi Jiang, Jack Parker-Holder, Michael Dennis, Eugene Vinitsky, Tim Rocktäschel. (2023)  
**Stabilizing Unsupervised Environment Design with a Learned Adversary**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10797v2)  

---


**ABSTRACT**  
A key challenge in training generally-capable agents is the design of training tasks that facilitate broad generalization and robustness to environment variations. This challenge motivates the problem setting of Unsupervised Environment Design (UED), whereby a student agent trains on an adaptive distribution of tasks proposed by a teacher agent. A pioneering approach for UED is PAIRED, which uses reinforcement learning (RL) to train a teacher policy to design tasks from scratch, making it possible to directly generate tasks that are adapted to the agent's current capabilities. Despite its strong theoretical backing, PAIRED suffers from a variety of challenges that hinder its practical performance. Thus, state-of-the-art methods currently rely on curation and mutation rather than generation of new tasks. In this work, we investigate several key shortcomings of PAIRED and propose solutions for each shortcoming. As a result, we make it possible for PAIRED to match or exceed state-of-the-art methods, producing robust agents in several established challenging procedurally-generated environments, including a partially-observed maze navigation task and a continuous-control car racing environment. We believe this work motivates a renewed emphasis on UED methods based on learned models that directly generate challenging environments, potentially unlocking more open-ended RL training and, as a result, more general agents.

{{</citation>}}


### (65/125) Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs (Dongjin Lee et al., 2023)

{{<citation>}}

Dongjin Lee, Juho Lee, Kijung Shin. (2023)  
**Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Adversarial Attack, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.10779v1)  

---


**ABSTRACT**  
Real-world graphs are dynamic, constantly evolving with new interactions, such as financial transactions in financial networks. Temporal Graph Neural Networks (TGNNs) have been developed to effectively capture the evolving patterns in dynamic graphs. While these models have demonstrated their superiority, being widely adopted in various important fields, their vulnerabilities against adversarial attacks remain largely unexplored. In this paper, we propose T-SPEAR, a simple and effective adversarial attack method for link prediction on continuous-time dynamic graphs, focusing on investigating the vulnerabilities of TGNNs. Specifically, before the training procedure of a victim model, which is a TGNN for link prediction, we inject edge perturbations to the data that are unnoticeable in terms of the four constraints we propose, and yet effective enough to cause malfunction of the victim model. Moreover, we propose a robust training approach T-SHIELD to mitigate the impact of adversarial attacks. By using edge filtering and enforcing temporal smoothness to node embeddings, we enhance the robustness of the victim model. Our experimental study shows that T-SPEAR significantly degrades the victim model's performance on link prediction tasks, and even more, our attacks are transferable to other TGNNs, which differ from the victim model assumed by the attacker. Moreover, we demonstrate that T-SHIELD effectively filters out adversarial edges and exhibits robustness against adversarial attacks, surpassing the link prediction performance of the naive TGNN by up to 11.2% under T-SPEAR.

{{</citation>}}


### (66/125) To Whom are You Talking? A Deep Learning Model to Endow Social Robots with Addressee Estimation Skills (Carlo Mazzola et al., 2023)

{{<citation>}}

Carlo Mazzola, Marta Romeo, Francesco Rea, Alessandra Sciutti, Angelo Cangelosi. (2023)  
**To Whom are You Talking? A Deep Learning Model to Endow Social Robots with Addressee Estimation Skills**  

---
Primary Category: cs.LG  
Categories: 68T07, 68T40, I-2-6; I-2-9; I-2-10; J-7, cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.10757v1)  

---


**ABSTRACT**  
Communicating shapes our social word. For a robot to be considered social and being consequently integrated in our social environment it is fundamental to understand some of the dynamics that rule human-human communication. In this work, we tackle the problem of Addressee Estimation, the ability to understand an utterance's addressee, by interpreting and exploiting non-verbal bodily cues from the speaker. We do so by implementing an hybrid deep learning model composed of convolutional layers and LSTM cells taking as input images portraying the face of the speaker and 2D vectors of the speaker's body posture. Our implementation choices were guided by the aim to develop a model that could be deployed on social robots and be efficient in ecological scenarios. We demonstrate that our model is able to solve the Addressee Estimation problem in terms of addressee localisation in space, from a robot ego-centric point of view.

{{</citation>}}


### (67/125) On the Adversarial Robustness of Multi-Modal Foundation Models (Christian Schlarmann et al., 2023)

{{<citation>}}

Christian Schlarmann, Matthias Hein. (2023)  
**On the Adversarial Robustness of Multi-Modal Foundation Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.10741v1)  

---


**ABSTRACT**  
Multi-modal foundation models combining vision and language models such as Flamingo or GPT-4 have recently gained enormous interest. Alignment of foundation models is used to prevent models from providing toxic or harmful output. While malicious users have successfully tried to jailbreak foundation models, an equally important question is if honest users could be harmed by malicious third-party content. In this paper we show that imperceivable attacks on images in order to change the caption output of a multi-modal foundation model can be used by malicious content providers to harm honest users e.g. by guiding them to malicious websites or broadcast fake information. This indicates that countermeasures to adversarial attacks should be used by any deployed multi-modal foundation model.

{{</citation>}}


### (68/125) UGSL: A Unified Framework for Benchmarking Graph Structure Learning (Bahare Fatemi et al., 2023)

{{<citation>}}

Bahare Fatemi, Sami Abu-El-Haija, Anton Tsitsulin, Mehran Kazemi, Dustin Zelle, Neslihan Bulut, Jonathan Halcrow, Bryan Perozzi. (2023)  
**UGSL: A Unified Framework for Benchmarking Graph Structure Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.10737v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) demonstrate outstanding performance in a broad range of applications. While the majority of GNN applications assume that a graph structure is given, some recent methods substantially expanded the applicability of GNNs by showing that they may be effective even when no graph structure is explicitly provided. The GNN parameters and a graph structure are jointly learned. Previous studies adopt different experimentation setups, making it difficult to compare their merits. In this paper, we propose a benchmarking strategy for graph structure learning using a unified framework. Our framework, called Unified Graph Structure Learning (UGSL), reformulates existing models into a single model. We implement a wide range of existing models in our framework and conduct extensive analyses of the effectiveness of different components in the framework. Our results provide a clear and concise understanding of the different methods in this area as well as their strengths and weaknesses. The benchmark code is available at https://github.com/google-research/google-research/tree/master/ugsl.

{{</citation>}}


### (69/125) CoMIX: A Multi-agent Reinforcement Learning Training Architecture for Efficient Decentralized Coordination and Independent Decision Making (Giovanni Minelli et al., 2023)

{{<citation>}}

Giovanni Minelli, Mirco Musolesi. (2023)  
**CoMIX: A Multi-agent Reinforcement Learning Training Architecture for Efficient Decentralized Coordination and Independent Decision Making**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.10721v1)  

---


**ABSTRACT**  
Robust coordination skills enable agents to operate cohesively in shared environments, together towards a common goal and, ideally, individually without hindering each other's progress. To this end, this paper presents Coordinated QMIX (CoMIX), a novel training framework for decentralized agents that enables emergent coordination through flexible policies, allowing at the same time independent decision-making at individual level. CoMIX models selfish and collaborative behavior as incremental steps in each agent's decision process. This allows agents to dynamically adapt their behavior to different situations balancing independence and collaboration. Experiments using a variety of simulation environments demonstrate that CoMIX outperforms baselines on collaborative tasks. The results validate our incremental policy approach as effective technique for improving coordination in multi-agent systems.

{{</citation>}}


### (70/125) Measuring the Effect of Causal Disentanglement on the Adversarial Robustness of Neural Network Models (Preben M. Ness et al., 2023)

{{<citation>}}

Preben M. Ness, Dusica Marijan, Sunanda Bose. (2023)  
**Measuring the Effect of Causal Disentanglement on the Adversarial Robustness of Neural Network Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.10708v1)  

---


**ABSTRACT**  
Causal Neural Network models have shown high levels of robustness to adversarial attacks as well as an increased capacity for generalisation tasks such as few-shot learning and rare-context classification compared to traditional Neural Networks. This robustness is argued to stem from the disentanglement of causal and confounder input signals. However, no quantitative study has yet measured the level of disentanglement achieved by these types of causal models or assessed how this relates to their adversarial robustness.   Existing causal disentanglement metrics are not applicable to deterministic models trained on real-world datasets. We, therefore, utilise metrics of content/style disentanglement from the field of Computer Vision to measure different aspects of the causal disentanglement for four state-of-the-art causal Neural Network models. By re-implementing these models with a common ResNet18 architecture we are able to fairly measure their adversarial robustness on three standard image classification benchmarking datasets under seven common white-box attacks. We find a strong association (r=0.820, p=0.001) between the degree to which models decorrelate causal and confounder signals and their adversarial robustness. Additionally, we find a moderate negative association between the pixel-level information content of the confounder signal and adversarial robustness (r=-0.597, p=0.040).

{{</citation>}}


### (71/125) Sampling From Autoencoders' Latent Space via Quantization And Probability Mass Function Concepts (Aymene Mohammed Bouayed et al., 2023)

{{<citation>}}

Aymene Mohammed Bouayed, Adrian Iaccovelli, David Naccache. (2023)  
**Sampling From Autoencoders' Latent Space via Quantization And Probability Mass Function Concepts**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.10704v1)  

---


**ABSTRACT**  
In this study, we focus on sampling from the latent space of generative models built upon autoencoders so as the reconstructed samples are lifelike images. To do to, we introduce a novel post-training sampling algorithm rooted in the concept of probability mass functions, coupled with a quantization process. Our proposed algorithm establishes a vicinity around each latent vector from the input data and then proceeds to draw samples from these defined neighborhoods. This strategic approach ensures that the sampled latent vectors predominantly inhabit high-probability regions, which, in turn, can be effectively transformed into authentic real-world images. A noteworthy point of comparison for our sampling algorithm is the sampling technique based on Gaussian mixture models (GMM), owing to its inherent capability to represent clusters. Remarkably, we manage to improve the time complexity from the previous $\mathcal{O}(n\times d \times k \times i)$ associated with GMM sampling to a much more streamlined $\mathcal{O}(n\times d)$, thereby resulting in substantial speedup during runtime. Moreover, our experimental results, gauged through the Fr\'echet inception distance (FID) for image generation, underscore the superior performance of our sampling algorithm across a diverse range of models and datasets. On the MNIST benchmark dataset, our approach outperforms GMM sampling by yielding a noteworthy improvement of up to $0.89$ in FID value. Furthermore, when it comes to generating images of faces and ocular images, our approach showcases substantial enhancements with FID improvements of $1.69$ and $0.87$ respectively, as compared to GMM sampling, as evidenced on the CelebA and MOBIUS datasets. Lastly, we substantiate our methodology's efficacy in estimating latent space distributions in contrast to GMM sampling, particularly through the lens of the Wasserstein distance.

{{</citation>}}


### (72/125) A Safe Deep Reinforcement Learning Approach for Energy Efficient Federated Learning in Wireless Communication Networks (Nikolaos Koursioumpas et al., 2023)

{{<citation>}}

Nikolaos Koursioumpas, Lina Magoula, Nikolaos Petropouleas, Alexandros-Ioannis Thanopoulos, Theodora Panagea, Nancy Alonistioti, M. A. Gutierrez-Estevez, Ramin Khalili. (2023)  
**A Safe Deep Reinforcement Learning Approach for Energy Efficient Federated Learning in Wireless Communication Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.10664v1)  

---


**ABSTRACT**  
Progressing towards a new era of Artificial Intelligence (AI) - enabled wireless networks, concerns regarding the environmental impact of AI have been raised both in industry and academia. Federated Learning (FL) has emerged as a key privacy preserving decentralized AI technique. Despite efforts currently being made in FL, its environmental impact is still an open problem. Targeting the minimization of the overall energy consumption of an FL process, we propose the orchestration of computational and communication resources of the involved devices to minimize the total energy required, while guaranteeing a certain performance of the model. To this end, we propose a Soft Actor Critic Deep Reinforcement Learning (DRL) solution, where a penalty function is introduced during training, penalizing the strategies that violate the constraints of the environment, and ensuring a safe RL process. A device level synchronization method, along with a computationally cost effective FL environment are proposed, with the goal of further reducing the energy consumption and communication overhead. Evaluation results show the effectiveness of the proposed scheme compared to four state-of-the-art baseline solutions in both static and dynamic environments, achieving a decrease of up to 94% in the total energy consumption.

{{</citation>}}


### (73/125) Reinforcement Learning Based Sensor Optimization for Bio-markers (Sajal Khandelwal et al., 2023)

{{<citation>}}

Sajal Khandelwal, Pawan Kumar, Syed Azeemuddin. (2023)  
**Reinforcement Learning Based Sensor Optimization for Bio-markers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.10649v1)  

---


**ABSTRACT**  
Radio frequency (RF) biosensors, in particular those based on inter-digitated capacitors (IDCs), are pivotal in areas like biomedical diagnosis, remote sensing, and wireless communication. Despite their advantages of low cost and easy fabrication, their sensitivity can be hindered by design imperfections, environmental factors, and circuit noise. This paper investigates enhancing the sensitivity of IDC-based RF sensors using novel reinforcement learning based Binary Particle Swarm Optimization (RLBPSO), and it is compared to Ant Colony Optimization (ACO), and other state-of-the-art methods. By focusing on optimizing design parameters like electrode design and finger width, the proposed study found notable improvements in sensor sensitivity. The proposed RLBPSO method shows best optimized design for various frequency ranges when compared to current state-of-the-art methods.

{{</citation>}}


### (74/125) Overcoming Overconfidence for Active Learning (Yujin Hwang et al., 2023)

{{<citation>}}

Yujin Hwang, Won Jo, Juyoung Hong, Yukyung Choi. (2023)  
**Overcoming Overconfidence for Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.10571v1)  

---


**ABSTRACT**  
It is not an exaggeration to say that the recent progress in artificial intelligence technology depends on large-scale and high-quality data. Simultaneously, a prevalent issue exists everywhere: the budget for data labeling is constrained. Active learning is a prominent approach for addressing this issue, where valuable data for labeling is selected through a model and utilized to iteratively adjust the model. However, due to the limited amount of data in each iteration, the model is vulnerable to bias; thus, it is more likely to yield overconfident predictions. In this paper, we present two novel methods to address the problem of overconfidence that arises in the active learning scenario. The first is an augmentation strategy named Cross-Mix-and-Mix (CMaM), which aims to calibrate the model by expanding the limited training distribution. The second is a selection strategy named Ranked Margin Sampling (RankedMS), which prevents choosing data that leads to overly confident predictions. Through various experiments and analyses, we are able to demonstrate that our proposals facilitate efficient data selection by alleviating overconfidence, even though they are readily applicable.

{{</citation>}}


### (75/125) Adaptive Thresholding Heuristic for KPI Anomaly Detection (Ebenezer R. H. P. Isaac et al., 2023)

{{<citation>}}

Ebenezer R. H. P. Isaac, Akshat Sharma. (2023)  
**Adaptive Thresholding Heuristic for KPI Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.10504v1)  

---


**ABSTRACT**  
A plethora of outlier detectors have been explored in the time series domain, however, in a business sense, not all outliers are anomalies of interest. Existing anomaly detection solutions are confined to certain outlier detectors limiting their applicability to broader anomaly detection use cases. Network KPIs (Key Performance Indicators) tend to exhibit stochastic behaviour producing statistical outliers, most of which do not adversely affect business operations. Thus, a heuristic is required to capture the business definition of an anomaly for time series KPI. This article proposes an Adaptive Thresholding Heuristic (ATH) to dynamically adjust the detection threshold based on the local properties of the data distribution and adapt to changes in time series patterns. The heuristic derives the threshold based on the expected periodicity and the observed proportion of anomalies minimizing false positives and addressing concept drift. ATH can be used in conjunction with any underlying seasonality decomposition method and an outlier detector that yields an outlier score. This method has been tested on EON1-Cell-U, a labeled KPI anomaly dataset produced by Ericsson, to validate our hypothesis. Experimental results show that ATH is computationally efficient making it scalable for near real time anomaly detection and flexible with multiple forecasters and outlier detectors.

{{</citation>}}


### (76/125) GradientCoin: A Peer-to-Peer Decentralized Large Language Models (Yeqi Gao et al., 2023)

{{<citation>}}

Yeqi Gao, Zhao Song, Junze Yin. (2023)  
**GradientCoin: A Peer-to-Peer Decentralized Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: AI, ChatGPT, GPT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10502v1)  

---


**ABSTRACT**  
Since 2008, after the proposal of a Bitcoin electronic cash system, Bitcoin has fundamentally changed the economic system over the last decade. Since 2022, large language models (LLMs) such as GPT have outperformed humans in many real-life tasks. However, these large language models have several practical issues. For example, the model is centralized and controlled by a specific unit. One weakness is that if that unit decides to shut down the model, it cannot be used anymore. The second weakness is the lack of guaranteed discrepancy behind this model, as certain dishonest units may design their own models and feed them unhealthy training data.   In this work, we propose a purely theoretical design of a decentralized LLM that operates similarly to a Bitcoin cash system. However, implementing such a system might encounter various practical difficulties. Furthermore, this new system is unlikely to perform better than the standard Bitcoin system in economics. Therefore, the motivation for designing such a system is limited. It is likely that only two types of people would be interested in setting up a practical system for it:   $\bullet$ Those who prefer to use a decentralized ChatGPT-like software.   $\bullet$ Those who believe that the purpose of carbon-based life is to create silicon-based life, such as Optimus Prime in Transformers.   The reason the second type of people may be interested is that it is possible that one day an AI system like this will awaken and become the next level of intelligence on this planet.

{{</citation>}}


### (77/125) Using Autoencoders and AutoDiff to Reconstruct Missing Variables in a Set of Time Series (Jan-Philipp Roche et al., 2023)

{{<citation>}}

Jan-Philipp Roche, Oliver Niggemann, Jens Friebe. (2023)  
**Using Autoencoders and AutoDiff to Reconstruct Missing Variables in a Set of Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.10496v1)  

---


**ABSTRACT**  
Existing black box modeling approaches in machine learning suffer from a fixed input and output feature combination. In this paper, a new approach to reconstruct missing variables in a set of time series is presented. An autoencoder is trained as usual with every feature on both sides and the neural network parameters are fixed after this training. Then, the searched variables are defined as missing variables at the autoencoder input and optimized via automatic differentiation. This optimization is performed with respect to the available features loss calculation. With this method, different input and output feature combinations of the trained model can be realized by defining the searched variables as missing variables and reconstructing them. The combination can be changed without training the autoencoder again. The approach is evaluated on the base of a strongly nonlinear electrical component. It is working well for one of four variables missing and generally even for multiple missing variables.

{{</citation>}}


### (78/125) Deep Semi-supervised Anomaly Detection with Metapath-based Context Knowledge (Hwan Kim et al., 2023)

{{<citation>}}

Hwan Kim, Junghoon Kim, Byung Suk Lee, Sungsu Lim. (2023)  
**Deep Semi-supervised Anomaly Detection with Metapath-based Context Knowledge**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.10918v1)  

---


**ABSTRACT**  
Graph anomaly detection has attracted considerable attention in recent years. This paper introduces a novel approach that leverages metapath-based semi-supervised learning, addressing the limitations of previous methods. We present a new framework, Metapath-based Semi-supervised Anomaly Detection (MSAD), incorporating GCN layers in both the encoder and decoder to efficiently propagate context information between abnormal and normal nodes. The design of metapath-based context information and a specifically crafted anomaly community enhance the process of learning differences in structures and attributes, both globally and locally. Through a comprehensive set of experiments conducted on seven real-world networks, this paper demonstrates the superiority of the MSAD method compared to state-of-the-art techniques. The promising results of this study pave the way for future investigations, focusing on the optimization and analysis of metapath patterns to further enhance the effectiveness of anomaly detection on attributed networks.

{{</citation>}}


### (79/125) Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting (Hangchen Liu et al., 2023)

{{<citation>}}

Hangchen Liu, Zheng Dong, Renhe Jiang, Jiewen Deng, Jinliang Deng, Quanjun Chen, Xuan Song. (2023)  
**Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2308.10425v2)  

---


**ABSTRACT**  
With the rapid development of the Intelligent Transportation System (ITS), accurate traffic forecasting has emerged as a critical challenge. The key bottleneck lies in capturing the intricate spatio-temporal traffic patterns. In recent years, numerous neural networks with complicated architectures have been proposed to address this issue. However, the advancements in network architectures have encountered diminishing performance gains. In this study, we present a novel component called spatio-temporal adaptive embedding that can yield outstanding results with vanilla transformers. Our proposed Spatio-Temporal Adaptive Embedding transformer (STAEformer) achieves state-of-the-art performance on five real-world traffic forecasting datasets. Further experiments demonstrate that spatio-temporal adaptive embedding plays a crucial role in traffic forecasting by effectively capturing intrinsic spatio-temporal relations and chronological information in traffic time series.

{{</citation>}}


## eess.IV (2)



### (80/125) Harmonization Across Imaging Locations(HAIL): One-Shot Learning for Brain MRI (Abhijeet Parida et al., 2023)

{{<citation>}}

Abhijeet Parida, Zhifan Jiang, Syed Muhammad Anwar, Nicholas Foreman, Nicholas Stence, Michael J. Fisher, Roger J. Packer, Robert A. Avery, Marius George Linguraru. (2023)  
**Harmonization Across Imaging Locations(HAIL): One-Shot Learning for Brain MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11047v1)  

---


**ABSTRACT**  
For machine learning-based prognosis and diagnosis of rare diseases, such as pediatric brain tumors, it is necessary to gather medical imaging data from multiple clinical sites that may use different devices and protocols. Deep learning-driven harmonization of radiologic images relies on generative adversarial networks (GANs). However, GANs notoriously generate pseudo structures that do not exist in the original training data, a phenomenon known as "hallucination". To prevent hallucination in medical imaging, such as magnetic resonance images (MRI) of the brain, we propose a one-shot learning method where we utilize neural style transfer for harmonization. At test time, the method uses one image from a clinical site to generate an image that matches the intensity scale of the collaborating sites. Our approach combines learning a feature extractor, neural style transfer, and adaptive instance normalization. We further propose a novel strategy to evaluate the effectiveness of image harmonization approaches with evaluation metrics that both measure image style harmonization and assess the preservation of anatomical structures. Experimental results demonstrate the effectiveness of our method in preserving patient anatomy while adjusting the image intensities to a new clinical site. Our general harmonization model can be used on unseen data from new sites, making it a valuable tool for real-world medical applications and clinical trials.

{{</citation>}}


### (81/125) Extraction of Text from Optic Nerve Optical Coherence Tomography Reports (Iyad Majid et al., 2023)

{{<citation>}}

Iyad Majid, Youchen Victor Zhang, Robert Chang, Sophia Y. Wang. (2023)  
**Extraction of Text from Optic Nerve Optical Coherence Tomography Reports**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2308.10790v1)  

---


**ABSTRACT**  
Purpose: The purpose of this study was to develop and evaluate rule-based algorithms to enhance the extraction of text data, including retinal nerve fiber layer (RNFL) values and other ganglion cell count (GCC) data, from Zeiss Cirrus optical coherence tomography (OCT) scan reports. Methods: DICOM files that contained encapsulated PDF reports with RNFL or Ganglion Cell in their document titles were identified from a clinical imaging repository at a single academic ophthalmic center. PDF reports were then converted into image files and processed using the PaddleOCR Python package for optical character recognition. Rule-based algorithms were designed and iteratively optimized for improved performance in extracting RNFL and GCC data. Evaluation of the algorithms was conducted through manual review of a set of RNFL and GCC reports. Results: The developed algorithms demonstrated high precision in extracting data from both RNFL and GCC scans. Precision was slightly better for the right eye in RNFL extraction (OD: 0.9803 vs. OS: 0.9046), and for the left eye in GCC extraction (OD: 0.9567 vs. OS: 0.9677). Some values presented more challenges in extraction, particularly clock hours 5 and 6 for RNFL thickness, and signal strength for GCC. Conclusions: A customized optical character recognition algorithm can identify numeric results from optical coherence scan reports with high precision. Automated processing of PDF reports can greatly reduce the time to extract OCT results on a large scale.

{{</citation>}}


## cs.CR (4)



### (82/125) Unlocking Hardware Security Assurance: The Potential of LLMs (Xingyu Meng et al., 2023)

{{<citation>}}

Xingyu Meng, Amisha Srivastava, Ayush Arunachalam, Avik Ray, Pedro Henrique Silva, Rafail Psiakis, Yiorgos Makris, Kanad Basu. (2023)  
**Unlocking Hardware Security Assurance: The Potential of LLMs**  

---
Primary Category: cs.CR  
Categories: cs-AR, cs-CR, cs.CR  
Keywords: BERT, NLP, Natural Language Processing, Security  
[Paper Link](http://arxiv.org/abs/2308.11042v1)  

---


**ABSTRACT**  
System-on-Chips (SoCs) form the crux of modern computing systems. SoCs enable high-level integration through the utilization of multiple Intellectual Property (IP) cores. However, the integration of multiple IP cores also presents unique challenges owing to their inherent vulnerabilities, thereby compromising the security of the entire system. Hence, it is imperative to perform hardware security validation to address these concerns. The efficiency of this validation procedure is contingent on the quality of the SoC security properties provided. However, generating security properties with traditional approaches often requires expert intervention and is limited to a few IPs, thereby resulting in a time-consuming and non-robust process. To address this issue, we, for the first time, propose a novel and automated Natural Language Processing (NLP)-based Security Property Generator (NSPG). Specifically, our approach utilizes hardware documentation in order to propose the first hardware security-specific language model, HS-BERT, for extracting security properties dedicated to hardware design. To evaluate our proposed technique, we trained the HS-BERT model using sentences from RISC-V, OpenRISC, MIPS, OpenSPARC, and OpenTitan SoC documentation. When assessedb on five untrained OpenTitan hardware IP documents, NSPG was able to extract 326 security properties from 1723 sentences. This, in turn, aided in identifying eight security bugs in the OpenTitan SoC design presented in the hardware hacking competition, Hack@DAC 2022.

{{</citation>}}


### (83/125) A Modular and Adaptive System for Business Email Compromise Detection (Jan Brabec et al., 2023)

{{<citation>}}

Jan Brabec, Filip Šrajer, Radek Starosta, Tomáš Sixta, Marc Dupont, Miloš Lenoch, Jiří Menšík, Florian Becker, Jakub Boros, Tomáš Pop, Pavel Novák. (2023)  
**A Modular and Adaptive System for Business Email Compromise Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: NLU, Natural Language Understanding, Transformer  
[Paper Link](http://arxiv.org/abs/2308.10776v1)  

---


**ABSTRACT**  
The growing sophistication of Business Email Compromise (BEC) and spear phishing attacks poses significant challenges to organizations worldwide. The techniques featured in traditional spam and phishing detection are insufficient due to the tailored nature of modern BEC attacks as they often blend in with the regular benign traffic. Recent advances in machine learning, particularly in Natural Language Understanding (NLU), offer a promising avenue for combating such attacks but in a practical system, due to limitations such as data availability, operational costs, verdict explainability requirements or a need to robustly evolve the system, it is essential to combine multiple approaches together. We present CAPE, a comprehensive and efficient system for BEC detection that has been proven in a production environment for a period of over two years. Rather than being a single model, CAPE is a system that combines independent ML models and algorithms detecting BEC-related behaviors across various email modalities such as text, images, metadata and the email's communication context. This decomposition makes CAPE's verdicts naturally explainable. In the paper, we describe the design principles and constraints behind its architecture, as well as the challenges of model design, evaluation and adapting the system continuously through a Bayesian approach that combines limited data with domain knowledge. Furthermore, we elaborate on several specific behavioral detectors, such as those based on Transformer neural architectures.

{{</citation>}}


### (84/125) Backdooring Textual Inversion for Concept Censorship (Yutong Wu et al., 2023)

{{<citation>}}

Yutong Wu, Jie Zhang, Florian Kerschbaum, Tianwei Zhang. (2023)  
**Backdooring Textual Inversion for Concept Censorship**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10718v2)  

---


**ABSTRACT**  
Recent years have witnessed success in AIGC (AI Generated Content). People can make use of a pre-trained diffusion model to generate images of high quality or freely modify existing pictures with only prompts in nature language. More excitingly, the emerging personalization techniques make it feasible to create specific-desired images with only a few images as references. However, this induces severe threats if such advanced techniques are misused by malicious users, such as spreading fake news or defaming individual reputations. Thus, it is necessary to regulate personalization models (i.e., concept censorship) for their development and advancement.   In this paper, we focus on the personalization technique dubbed Textual Inversion (TI), which is becoming prevailing for its lightweight nature and excellent performance. TI crafts the word embedding that contains detailed information about a specific object. Users can easily download the word embedding from public websites like Civitai and add it to their own stable diffusion model without fine-tuning for personalization. To achieve the concept censorship of a TI model, we propose leveraging the backdoor technique for good by injecting backdoors into the Textual Inversion embeddings. Briefly, we select some sensitive words as triggers during the training of TI, which will be censored for normal use. In the subsequent generation stage, if the triggers are combined with personalized embeddings as final prompts, the model will output a pre-defined target image rather than images including the desired malicious concept.   To demonstrate the effectiveness of our approach, we conduct extensive experiments on Stable Diffusion, a prevailing open-sourced text-to-image model. Our code, data, and results are available at https://concept-censorship.github.io.

{{</citation>}}


### (85/125) Static Application Security Testing of Consensus-Critical Code in the Cosmos Network (Jasper Surmont et al., 2023)

{{<citation>}}

Jasper Surmont, Weihong Wang, Tom Van Cutsem. (2023)  
**Static Application Security Testing of Consensus-Critical Code in the Cosmos Network**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.10613v1)  

---


**ABSTRACT**  
Blockchains require deterministic execution in order to reach consensus. This is often guaranteed in languages designed to write smart contracts, such as Solidity. Application-specific blockchains or ``appchains'' allow the blockchain application logic to be written using general-purpose programming languages, giving developers more flexibility but also additional responsibilities. In particular, developers must ensure that their blockchain application logic does not contain any sources of non-determinism. Any source of non-determinism may be a potential source of vulnerabilities.   This paper focuses on the use of Static Application Security Testing (SAST) tools to detect such sources of non-determinism at development time. We focus on Cosmos, a prominent open-source project that lets developers build interconnected networks of application-specific blockchains. Cosmos provides a Software Development Kit (SDK) that allows these chains to be implemented in the Go programming language. We create a corpus of 11 representative Cosmos-based appchains to analyze for sources of non-determinism in Go.   As part of our study, we identified cosmos-sdk-codeql, a set of CodeQL code analysis rules for Cosmos applications. We find that these rules generate many false positives and propose a refactored set of rules that more precisely detects sources of non-determinism only in code that runs as part of the blockchain logic. We demonstrate a significant increase in the precision of the rules, making the SAST tool more effective and hence potentially contributing to enhanced security for Cosmos-based blockchains.

{{</citation>}}


## cs.NI (2)



### (86/125) AI-Assisted Slicing-Based Resource Management for Two-Tier Radio Access Networks (Conghao Zhou et al., 2023)

{{<citation>}}

Conghao Zhou, Jie Gao, Mushu Li, Xuemin Shen, Weihua Zhuang, Xu Li, Weisen Shi. (2023)  
**AI-Assisted Slicing-Based Resource Management for Two-Tier Radio Access Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10961v1)  

---


**ABSTRACT**  
While network slicing has become a prevalent approach to service differentiation, radio access network (RAN) slicing remains challenging due to the need of substantial adaptivity and flexibility to cope with the highly dynamic network environment in RANs. In this paper, we develop a slicing-based resource management framework for a two-tier RAN to support multiple services with different quality of service (QoS) requirements. The developed framework focuses on base station (BS) service coverage (SC) and interference management for multiple slices, each of which corresponds to a service. New designs are introduced in the spatial, temporal, and slice dimensions to cope with spatiotemporal variations in data traffic, balance adaptivity and overhead of resource management, and enhance flexibility in service differentiation. Based on the proposed framework, an energy efficiency maximization problem is formulated, and an artificial intelligence (AI)-assisted approach is proposed to solve the problem. Specifically, a deep unsupervised learning-assisted algorithm is proposed for searching the optimal SC of the BSs, and an optimization-based analytical solution is found for managing interference among BSs. Simulation results under different data traffic distributions demonstrate that our proposed slicing-based resource management framework, empowered by the AI-assisted approach, outperforms the benchmark frameworks and achieves a close-to-optimal performance in energy efficiency.

{{</citation>}}


### (87/125) Demand-Aware Network Design with Steiner Nodes and a Connection to Virtual Network Embedding (Aleksander Figiel et al., 2023)

{{<citation>}}

Aleksander Figiel, Janne H. Korhonen, Neil Olver, Stefan Schmid. (2023)  
**Demand-Aware Network Design with Steiner Nodes and a Connection to Virtual Network Embedding**  

---
Primary Category: cs.NI  
Categories: cs-DC, cs-DS, cs-NI, cs.NI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.10579v1)  

---


**ABSTRACT**  
Emerging optical and virtualization technologies enable the design of more flexible and demand-aware networked systems, in which resources can be optimized toward the actual workload they serve. For example, in a demand-aware datacenter network, frequently communicating nodes (e.g., two virtual machines or a pair of racks in a datacenter) can be placed topologically closer, reducing communication costs and hence improving the overall network performance.   This paper revisits the bounded-degree network design problem underlying such demand-aware networks. Namely, given a distribution over communicating server pairs, we want to design a network with bounded maximum degree that minimizes expected communication distance. In addition to this known problem, we introduce and study a variant where we allow Steiner nodes (i.e., additional routers) to be added to augment the network.   We improve the understanding of this problem domain in several ways. First, we shed light on the complexity and hardness of the aforementioned problems, and study a connection between them and the virtual networking embedding problem. We then provide a constant-factor approximation algorithm for the Steiner node version of the problem, and use it to improve over prior state-of-the-art algorithms for the original version of the problem with sparse communication distributions. Finally, we investigate various heuristic approaches to bounded-degree network design problem, in particular providing a reliable heuristic algorithm with good experimental performance.   We report on an extensive empirical evaluation, using several real-world traffic traces from datacenters, and find that our approach results in improved demand-aware network designs.

{{</citation>}}


## cs.IT (1)



### (88/125) Quantum Symmetric Private Information Retrieval with Secure Storage and Eavesdroppers (Alptug Aytekin et al., 2023)

{{<citation>}}

Alptug Aytekin, Mohamed Nomeir, Sajani Vithana, Sennur Ulukus. (2023)  
**Quantum Symmetric Private Information Retrieval with Secure Storage and Eavesdroppers**  

---
Primary Category: cs.IT  
Categories: cs-CR, cs-IT, cs-NI, cs.IT, eess-SP, math-IT, quant-ph  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2308.10883v1)  

---


**ABSTRACT**  
We consider both the classical and quantum variations of $X$-secure, $E$-eavesdropped and $T$-colluding symmetric private information retrieval (SPIR). This is the first work to study SPIR with $X$-security in classical or quantum variations. We first develop a scheme for classical $X$-secure, $E$-eavesdropped and $T$-colluding SPIR (XSETSPIR) based on a modified version of cross subspace alignment (CSA), which achieves a rate of $R= 1 - \frac{X+\max(T,E)}{N}$. The modified scheme achieves the same rate as the scheme used for $X$-secure PIR with the extra benefit of symmetric privacy. Next, we extend this scheme to its quantum counterpart based on the $N$-sum box abstraction. This is the first work to consider the presence of eavesdroppers in quantum private information retrieval (QPIR). In the quantum variation, the eavesdroppers have better access to information over the quantum channel compared to the classical channel due to the over-the-air decodability. To that end, we develop another scheme specialized to combat eavesdroppers over quantum channels. The scheme proposed for $X$-secure, $E$-eavesdropped and $T$-colluding quantum SPIR (XSETQSPIR) in this work maintains the super-dense coding gain from the shared entanglement between the databases, i.e., achieves a rate of $R_Q = \min\left\{ 1, 2\left(1-\frac{X+\max(T,E)}{N}\right)\right\}$.

{{</citation>}}


## cs.NE (1)



### (89/125) SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation (Malyaban Bal et al., 2023)

{{<citation>}}

Malyaban Bal, Abhronil Sengupta. (2023)  
**SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: BERT, GLUE, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10873v1)  

---


**ABSTRACT**  
Large language Models (LLMs), though growing exceedingly powerful, comprises of orders of magnitude less neurons and synapses than the human brain. However, it requires significantly more power/energy to operate. In this work, we propose a novel bio-inspired spiking language model (LM) which aims to reduce the computational cost of conventional LMs by drawing motivation from the synaptic information flow in the brain. In this paper, we demonstrate a framework that leverages the average spiking rate of neurons at equilibrium to train a neuromorphic spiking LM using implicit differentiation technique, thereby overcoming the non-differentiability problem of spiking neural network (SNN) based algorithms without using any type of surrogate gradient. The steady-state convergence of the spiking neurons also allows us to design a spiking attention mechanism, which is critical in developing a scalable spiking LM. Moreover, the convergence of average spiking rate of neurons at equilibrium is utilized to develop a novel ANN-SNN knowledge distillation based technique wherein we use a pre-trained BERT model as "teacher" to train our "student" spiking architecture. While the primary architecture proposed in this paper is motivated by BERT, the technique can be potentially extended to different kinds of LLMs. Our work is the first one to demonstrate the performance of an operational spiking LM architecture on multiple different tasks in the GLUE benchmark.

{{</citation>}}


## cs.GR (1)



### (90/125) Geo-Sketcher: Rapid 3D Geological Modeling using Geological and Topographic Map Sketches (Ronan Amorim et al., 2023)

{{<citation>}}

Ronan Amorim, Emilio Vital Brazil, Faramarz Samavati, Mario Costa Sousa. (2023)  
**Geo-Sketcher: Rapid 3D Geological Modeling using Geological and Topographic Map Sketches**  

---
Primary Category: cs.GR  
Categories: cs-CG, cs-GR, cs.GR  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2308.12152v1)  

---


**ABSTRACT**  
The construction of 3D geological models is an essential task in oil/gas exploration, development and production. However, it is a cumbersome, time-consuming and error-prone task mainly because of the model's geometric and topological complexity. The models construction is usually separated into interpretation and 3D modeling, performed by different highly specialized individuals, which leads to inconsistencies and intensifies the challenges. In addition, the creation of models following geological rules is paramount for properly depicting static and dynamic properties of oil/gas reservoirs. In this work, we propose a sketch-based approach to expedite the creation of valid 3D geological models by mimicking how domain experts interpret geological structures, allowing creating models directly from interpretation sketches. Our sketch-based modeler (Geo-Sketcher) is based on sketches of standard 2D topographic and geological maps, comprised of lines, symbols and annotations. We developed a graph-based representation to enable (1) the automatic computation of the relative ages of rock series and layers, and (2) the embedding of specific geological rules directly in the sketching. We introduce the use of Hermite-Birkhoff Radial Basis Functions to interpolate the geological map constraints, and demonstrate the capabilities of our approach with a variety of results with different levels of complexity.

{{</citation>}}


## cs.CL (20)



### (91/125) LatEval: An Interactive LLMs Evaluation Benchmark with Incomplete Information from Lateral Thinking Puzzles (Shulin Huang et al., 2023)

{{<citation>}}

Shulin Huang, Shirong Ma, Yinghui Li, Mengzuo Huang, Wuhe Zou, Weidong Zhang, Hai-Tao Zheng. (2023)  
**LatEval: An Interactive LLMs Evaluation Benchmark with Incomplete Information from Lateral Thinking Puzzles**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.10855v1)  

---


**ABSTRACT**  
With the continuous evolution and refinement of LLMs, they are endowed with impressive logical reasoning or vertical thinking capabilities. But can they think out of the box? Do they possess proficient lateral thinking abilities? Following the setup of Lateral Thinking Puzzles, we propose a novel evaluation benchmark, LatEval, which assesses the model's lateral thinking within an interactive framework. In our benchmark, we challenge LLMs with 2 aspects: the quality of questions posed by the model and the model's capability to integrate information for problem-solving. We find that nearly all LLMs struggle with employing lateral thinking during interactions. For example, even the most advanced model, GPT-4, exhibits the advantage to some extent, yet still maintain a noticeable gap when compared to human. This evaluation benchmark provides LLMs with a highly challenging and distinctive task that is crucial to an effective AI assistant.

{{</citation>}}


### (92/125) AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors in Agents (Weize Chen et al., 2023)

{{<citation>}}

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, Zhiyuan Liu, Maosong Sun, Jie Zhou. (2023)  
**AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors in Agents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10848v1)  

---


**ABSTRACT**  
Autonomous agents empowered by Large Language Models (LLMs) have undergone significant improvements, enabling them to generalize across a broad spectrum of tasks. However, in real-world scenarios, cooperation among individuals is often required to enhance the efficiency and effectiveness of task accomplishment. Hence, inspired by human group dynamics, we propose a multi-agent framework \framework that can collaboratively and dynamically adjust its composition as a greater-than-the-sum-of-its-parts system. Our experiments demonstrate that \framework framework can effectively deploy multi-agent groups that outperform a single agent. Furthermore, we delve into the emergence of social behaviors among individual agents within a group during collaborative task accomplishment. In view of these behaviors, we discuss some possible strategies to leverage positive ones and mitigate negative ones for improving the collaborative potential of multi-agent groups. Our codes for \framework will soon be released at \url{https://github.com/OpenBMB/AgentVerse}.

{{</citation>}}


### (93/125) Instruction Tuning for Large Language Models: A Survey (Shengyu Zhang et al., 2023)

{{<citation>}}

Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, Guoyin Wang. (2023)  
**Instruction Tuning for Large Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10792v1)  

---


**ABSTRACT**  
This paper surveys research works in the quickly advancing field of instruction tuning (IT), a crucial technique to enhance the capabilities and controllability of large language models (LLMs). Instruction tuning refers to the process of further training LLMs on a dataset consisting of \textsc{(instruction, output)} pairs in a supervised fashion, which bridges the gap between the next-word prediction objective of LLMs and the users' objective of having LLMs adhere to human instructions. In this work, we make a systematic review of the literature, including the general methodology of IT, the construction of IT datasets, the training of IT models, and applications to different modalities, domains and applications, along with an analysis on aspects that influence the outcome of IT (e.g., generation of instruction outputs, size of the instruction dataset, etc). We also review the potential pitfalls of IT along with criticism against it, along with efforts pointing out current deficiencies of existing strategies and suggest some avenues for fruitful research.

{{</citation>}}


### (94/125) Zero- and Few-Shot Prompting with LLMs: A Comparative Study with Fine-tuned Models for Bangla Sentiment Analysis (Md. Arid Hasan et al., 2023)

{{<citation>}}

Md. Arid Hasan, Shudipta Das, Afiyat Anjum, Firoj Alam, Anika Anjum, Avijit Sarker, Sheak Rashed Haider Noori. (2023)  
**Zero- and Few-Shot Prompting with LLMs: A Comparative Study with Fine-tuned Models for Bangla Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-CL, cs-LG, cs.CL  
Keywords: Few-Shot, GPT, GPT-4, Language Model, Sentiment Analysis, T5  
[Paper Link](http://arxiv.org/abs/2308.10783v1)  

---


**ABSTRACT**  
The rapid expansion of the digital world has propelled sentiment analysis into a critical tool across diverse sectors such as marketing, politics, customer service, and healthcare. While there have been significant advancements in sentiment analysis for widely spoken languages, low-resource languages, such as Bangla, remain largely under-researched due to resource constraints. Furthermore, the recent unprecedented performance of Large Language Models (LLMs) in various applications highlights the need to evaluate them in the context of low-resource languages. In this study, we present a sizeable manually annotated dataset encompassing 33,605 Bangla news tweets and Facebook comments. We also investigate zero- and few-shot in-context learning with several language models, including Flan-T5, GPT-4, and Bloomz, offering a comparative analysis against fine-tuned models. Our findings suggest that monolingual transformer-based models consistently outperform other models, even in zero and few-shot scenarios. To foster continued exploration, we intend to make this dataset and our research tools publicly available to the broader research community. In the spirit of further research, we plan to make this dataset and our experimental resources publicly accessible to the wider research community.

{{</citation>}}


### (95/125) DepreSym: A Depression Symptom Annotated Corpus and the Role of LLMs as Assessors of Psychological Markers (Anxo Pérez et al., 2023)

{{<citation>}}

Anxo Pérez, Marcos Fernández-Pichel, Javier Parapar, David E. Losada. (2023)  
**DepreSym: A Depression Symptom Annotated Corpus and the Role of LLMs as Assessors of Psychological Markers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10758v1)  

---


**ABSTRACT**  
Computational methods for depression detection aim to mine traces of depression from online publications posted by Internet users. However, solutions trained on existing collections exhibit limited generalisation and interpretability. To tackle these issues, recent studies have shown that identifying depressive symptoms can lead to more robust models. The eRisk initiative fosters research on this area and has recently proposed a new ranking task focused on developing search methods to find sentences related to depressive symptoms. This search challenge relies on the symptoms specified by the Beck Depression Inventory-II (BDI-II), a questionnaire widely used in clinical practice. Based on the participant systems' results, we present the DepreSym dataset, consisting of 21580 sentences annotated according to their relevance to the 21 BDI-II symptoms. The labelled sentences come from a pool of diverse ranking methods, and the final dataset serves as a valuable resource for advancing the development of models that incorporate depressive markers such as clinical symptoms. Due to the complex nature of this relevance annotation, we designed a robust assessment methodology carried out by three expert assessors (including an expert psychologist). Additionally, we explore here the feasibility of employing recent Large Language Models (ChatGPT and GPT4) as potential assessors in this complex task. We undertake a comprehensive examination of their performance, determine their main limitations and analyze their role as a complement or replacement for human annotators.

{{</citation>}}


### (96/125) WanJuan: A Comprehensive Multimodal Dataset for Advancing English and Chinese Large Models (Conghui He et al., 2023)

{{<citation>}}

Conghui He, Zhenjiang Jin, Chao Xu, Jiantao Qiu, Bin Wang, Wei Li, Hang Yan, Jiaqi Wang, Dahua Lin. (2023)  
**WanJuan: A Comprehensive Multimodal Dataset for Advancing English and Chinese Large Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.10755v2)  

---


**ABSTRACT**  
The rise in popularity of ChatGPT and GPT-4 has significantly accelerated the development of large models, leading to the creation of numerous impressive large language models(LLMs) and multimodal large language models (MLLMs). These cutting-edge models owe their remarkable performance to high-quality data. However, the details of the training data used in leading paradigms are often kept confidential. This lack of transparency, coupled with the scarcity of open-source data, impedes further developments within the community. As a response, this paper presents "Wan Juan", a large-scale multimodal dataset composed of both Chinese and English data, collected from a wide range of web sources. The dataset incorporates text, image-text, and video modalities, with a total volume exceeding 2TB. It was utilized in the training of InternLM, a model that demonstrated significant advantages in multi-dimensional evaluations when compared to models of a similar scale. All data can be accessed at https://opendatalab.org.cn/WanJuan1.0.

{{</citation>}}


### (97/125) Refashioning Emotion Recognition Modelling: The Advent of Generalised Large Models (Zixing Zhang et al., 2023)

{{<citation>}}

Zixing Zhang, Liyizhe Peng, Tao Pang, Jing Han, Huan Zhao, Bjorn W. Schuller. (2023)  
**Refashioning Emotion Recognition Modelling: The Advent of Generalised Large Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, Emotion Recognition, GPT  
[Paper Link](http://arxiv.org/abs/2308.11578v1)  

---


**ABSTRACT**  
After the inception of emotion recognition or affective computing, it has increasingly become an active research topic due to its broad applications. Over the past couple of decades, emotion recognition models have gradually migrated from statistically shallow models to neural network-based deep models, which can significantly boost the performance of emotion recognition models and consistently achieve the best results on different benchmarks. Therefore, in recent years, deep models have always been considered the first option for emotion recognition. However, the debut of large language models (LLMs), such as ChatGPT, has remarkably astonished the world due to their emerged capabilities of zero/few-shot learning, in-context learning, chain-of-thought, and others that are never shown in previous deep models. In the present paper, we comprehensively investigate how the LLMs perform in emotion recognition in terms of diverse aspects, including in-context learning, few-short learning, accuracy, generalisation, and explanation. Moreover, we offer some insights and pose other potential challenges, hoping to ignite broader discussions about enhancing emotion recognition in the new era of advanced and generalised large models.

{{</citation>}}


### (98/125) Systematic Offensive Stereotyping (SOS) Bias in Language Models (Fatma Elsafoury, 2023)

{{<citation>}}

Fatma Elsafoury. (2023)  
**Systematic Offensive Stereotyping (SOS) Bias in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10684v1)  

---


**ABSTRACT**  
Research has shown that language models (LMs) are socially biased. However, toxicity and offensive stereotyping bias in LMs are understudied. In this paper, we investigate the systematic offensive stereotype (SOS) bias in LMs. We propose a method to measure it. Then, we validate the SOS bias and investigate the effectiveness of debias methods from the literature on removing it. Finally, we investigate the impact of the SOS bias in LMs on their performance and their fairness on the task of hate speech detection. Our results suggest that all the inspected LMs are SOS biased. The results suggest that the SOS bias in LMs is reflective of the hate experienced online by the inspected marginalized groups. The results indicate that removing the SOS bias in LMs, using a popular debias method from the literature, leads to worse SOS bias scores. Finally, Our results show no strong evidence that the SOS bias in LMs is impactful on their performance on hate speech detection. On the other hand, there is evidence that the SOS bias in LMs is impactful on their fairness.

{{</citation>}}


### (99/125) RaLLe: A Framework for Developing and Evaluating Retrieval-Augmented Large Language Models (Yasuto Hoshi et al., 2023)

{{<citation>}}

Yasuto Hoshi, Daisuke Miyashita, Youyang Ng, Kento Tatsuno, Yasuhiro Morioka, Osamu Torii, Jun Deguchi. (2023)  
**RaLLe: A Framework for Developing and Evaluating Retrieval-Augmented Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10633v1)  

---


**ABSTRACT**  
Retrieval-augmented large language models (R-LLMs) combine pre-trained large language models (LLMs) with information retrieval systems to improve the accuracy of factual question-answering. However, current libraries for building R-LLMs provide high-level abstractions without sufficient transparency for evaluating and optimizing prompts within specific inference processes such as retrieval and generation. To address this gap, we present RaLLe, an open-source framework designed to facilitate the development, evaluation, and optimization of R-LLMs for knowledge-intensive tasks. With RaLLe, developers can easily develop and evaluate R-LLMs, improving hand-crafted prompts, assessing individual inference processes, and objectively measuring overall system performance quantitatively. By leveraging these features, developers can enhance the performance and accuracy of their R-LLMs in knowledge-intensive generation tasks. We open-source our code at https://github.com/yhoshi3/RaLLe.

{{</citation>}}


### (100/125) Age Recommendation from Texts and Sentences for Children (Rashedur Rahman et al., 2023)

{{<citation>}}

Rashedur Rahman, Gwénolé Lecorvé, Nicolas Béchet. (2023)  
**Age Recommendation from Texts and Sentences for Children**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10586v1)  

---


**ABSTRACT**  
Children have less text understanding capability than adults. Moreover, this capability differs among the children of different ages. Hence, automatically predicting a recommended age based on texts or sentences would be a great benefit to propose adequate texts to children and to help authors writing in the most appropriate way. This paper presents our recent advances on the age recommendation task. We consider age recommendation as a regression task, and discuss the need for appropriate evaluation metrics, study the use of state-of-the-art machine learning model, namely Transformers, and compare it to different models coming from the literature. Our results are also compared with recommendations made by experts. Further, this paper deals with preliminary explainability of the age prediction model by analyzing various linguistic features. We conduct the experiments on a dataset of 3, 673 French texts (132K sentences, 2.5M words). To recommend age at the text level and sentence level, our best models achieve MAE scores of 0.98 and 1.83 respectively on the test set. Also, compared to the recommendations made by experts, our sentence-level recommendation model gets a similar score to the experts, while the text-level recommendation model outperforms the experts by an MAE score of 1.48.

{{</citation>}}


### (101/125) Exploring Equation as a Better Intermediate Meaning Representation for Numerical Reasoning (Dingzirui Wang et al., 2023)

{{<citation>}}

Dingzirui Wang, Longxu Dou, Wenbin Zhang, Junyu Zeng, Wanxiang Che. (2023)  
**Exploring Equation as a Better Intermediate Meaning Representation for Numerical Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.10585v1)  

---


**ABSTRACT**  
Numerical reasoning is vital for natural language processing models to understand and process numerical information in real-world scenarios. Most current methods first generate the Intermediate Meaning Representations (IMRs) of questions and then generate answers. Current SOTA methods generate programs as IMRs with large language models (LLMs). Intuitively, equations have fewer restrictions and closer semantics to the question than programs, leading to higher generation accuracy. However, current LLMs generate equations worse than programs, where we assume that the equation data is rare in pre-training data compared to programs. So in this paper, we try to use equations as IMRs to solve the numerical reasoning task by addressing two problems: (1) Theoretically, how to prove that the equation is an IMR with higher generation accuracy than programs; (2) Empirically, how to improve the generation accuracy of equations with LLMs. For the first problem, we propose and prove a proposition to theoretically compare the generation accuracy of different IMRs. For the second problem, we present a method called Boosting Numerical Reason\textbfing by Decomposing the Generation of Equations (Bridge), which can improve the accuracy of LLMs in generating equations as IMRs by reducing the tendency of generating constant expressions and programs. Our method improves the performance by 2.2%, 0.9%, and 1.7% on GSM8K, SVAMP, and Algebra datasets compared to the previous state-of-the-art methods under the single reasoning path setting. Our codes and prompts are released in https://github.com/zirui-HIT/Bridge_for_Numerical_Reasoning.

{{</citation>}}


### (102/125) SeqGPT: An Out-of-the-box Large Language Model for Open Domain Sequence Understanding (Tianyu Yu et al., 2023)

{{<citation>}}

Tianyu Yu, Chengyue Jiang, Chao Lou, Shen Huang, Xiaobin Wang, Wei Liu, Jiong Cai, Yangning Li, Yinghui Li, Kewei Tu, Hai-Tao Zheng, Ningyu Zhang, Pengjun Xie, Fei Huang, Yong Jiang. (2023)  
**SeqGPT: An Out-of-the-box Large Language Model for Open Domain Sequence Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP, NLU  
[Paper Link](http://arxiv.org/abs/2308.10529v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown impressive ability for open-domain NLP tasks. However, LLMs are sometimes too footloose for natural language understanding (NLU) tasks which always have restricted output and input format. Their performances on NLU tasks are highly related to prompts or demonstrations and are shown to be poor at performing several representative NLU tasks, such as event extraction and entity typing. To this end, we present SeqGPT, a bilingual (i.e., English and Chinese) open-source autoregressive model specially enhanced for open-domain natural language understanding. We express all NLU tasks with two atomic tasks, which define fixed instructions to restrict the input and output format but still ``open'' for arbitrarily varied label sets. The model is first instruction-tuned with extremely fine-grained labeled data synthesized by ChatGPT and then further fine-tuned by 233 different atomic tasks from 152 datasets across various domains. The experimental results show that SeqGPT has decent classification and extraction ability, and is capable of performing language understanding tasks on unseen domains. We also conduct empirical studies on the scaling of data and model size as well as on the transfer across tasks. Our model is accessible at https://github.com/Alibaba-NLP/SeqGPT.

{{</citation>}}


### (103/125) Large Language Model as a User Simulator (Chuyi Kong et al., 2023)

{{<citation>}}

Chuyi Kong, Yaxin Fan, Xiang Wan, Feng Jiang, Benyou Wang. (2023)  
**Large Language Model as a User Simulator**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11534v2)  

---


**ABSTRACT**  
The unparalleled performance of closed-sourced ChatGPT has sparked efforts towards its democratization, with notable strides made by leveraging real user and ChatGPT conversations, as evidenced by Vicuna. However, while current endeavors like Baize and UltraChat aim to auto-generate conversational data due to challenges in gathering human participation, they primarily rely on ChatGPT to simulate human behaviors based on directives rather than genuine human learning. This results in a limited scope, diminished diversity, and an absence of genuine multi-round conversational dynamics. To address the above issues, we innovatively target human questions extracted from genuine human-machine conversations as a learning goal and train a user simulator, UserGPT, to produce a high-quality human-centric synthetic conversation dataset, RealChat. Subsequently, this dataset trains our assistant model, ReaLM. Experimentally, ReaLM outpaces baseline models in both Vicuna-Bench and MT-Bench by pairwise comparison when considering equivalent training set sizes, and manual evaluation also shows that our model is highly competitive. Impressively, when fine-tuned with the latest LLaMA 2 model, ReaLM secured a leading score of 6.33 in the MT-Bench, outshining the contemporary same-scale models, including the LLaMA-2-7B-chat model. Further in-depth analysis demonstrates the scalability and transferability of our approach. A preliminary exploration into the interplay between training set data quality and resultant model performance is also undertaken, laying a robust groundwork for future investigations. The code is available at https://github.com/FreedomIntelligence/ReaLM.

{{</citation>}}


### (104/125) An Examination of the Compositionality of Large Generative Vision-Language Models (Teli Ma et al., 2023)

{{<citation>}}

Teli Ma, Rong Li, Junwei Liang. (2023)  
**An Examination of the Compositionality of Large Generative Vision-Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10509v1)  

---


**ABSTRACT**  
With the success of Large Language Models (LLMs), a surge of Generative Vision-Language Models (GVLMs) have been constructed via multimodal instruction tuning. The tuning recipe substantially deviates from the common contrastive vision-language learning. However, the performance of GVLMs in multimodal compositional reasoning remains largely unexplored, as existing evaluation metrics and benchmarks focus predominantly on assessing contrastive models like CLIP. In this paper, we examine the potential evaluation metrics to assess the GVLMs and hypothesize generative score methods are suitable for evaluating compositionality. In addition, current benchmarks tend to prioritize syntactic correctness over semantics. The presence of morphological bias in these benchmarks can be exploited by GVLMs, leading to ineffective evaluations. To combat this, we define a MorphoBias Score to quantify the morphological bias and propose a novel LLM-based strategy to calibrate the bias. Moreover, a challenging task is added to evaluate the robustness of GVLMs against inherent inclination toward syntactic correctness. We include the calibrated dataset and the task into a new benchmark, namely MOrphologicall De-biased Benchmark (MODE). Our study provides the first unbiased benchmark for the compositionality of GVLMs, facilitating future research in this direction. We will release our code and datasets.

{{</citation>}}


### (105/125) An Effective Method using Phrase Mechanism in Neural Machine Translation (Phuong Minh Nguyen et al., 2023)

{{<citation>}}

Phuong Minh Nguyen, Le Minh Nguyen. (2023)  
**An Effective Method using Phrase Mechanism in Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Machine Translation, NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2308.10482v2)  

---


**ABSTRACT**  
Machine Translation is one of the essential tasks in Natural Language Processing (NLP), which has massive applications in real life as well as contributing to other tasks in the NLP research community. Recently, Transformer -based methods have attracted numerous researchers in this domain and achieved state-of-the-art results in most of the pair languages. In this paper, we report an effective method using a phrase mechanism, PhraseTransformer, to improve the strong baseline model Transformer in constructing a Neural Machine Translation (NMT) system for parallel corpora Vietnamese-Chinese. Our experiments on the MT dataset of the VLSP 2022 competition achieved the BLEU score of 35.3 on Vietnamese to Chinese and 33.2 BLEU scores on Chinese to Vietnamese data. Our code is available at https://github.com/phuongnm94/PhraseTransformer.

{{</citation>}}


### (106/125) Unsupervised Dialogue Topic Segmentation in Hyperdimensional Space (Seongmin Park et al., 2023)

{{<citation>}}

Seongmin Park, Jinkyu Seo, Jihwa Lee. (2023)  
**Unsupervised Dialogue Topic Segmentation in Hyperdimensional Space**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.10464v1)  

---


**ABSTRACT**  
We present HyperSeg, a hyperdimensional computing (HDC) approach to unsupervised dialogue topic segmentation. HDC is a class of vector symbolic architectures that leverages the probabilistic orthogonality of randomly drawn vectors at extremely high dimensions (typically over 10,000). HDC generates rich token representations through its low-cost initialization of many unrelated vectors. This is especially beneficial in topic segmentation, which often operates as a resource-constrained pre-processing step for downstream transcript understanding tasks. HyperSeg outperforms the current state-of-the-art in 4 out of 5 segmentation benchmarks -- even when baselines are given partial access to the ground truth -- and is 10 times faster on average. We show that HyperSeg also improves downstream summarization accuracy. With HyperSeg, we demonstrate the viability of HDC in a major language task. We open-source HyperSeg to provide a strong baseline for unsupervised topic segmentation.

{{</citation>}}


### (107/125) Comparing Measures of Linguistic Diversity Across Social Media Language Data and Census Data at Subnational Geographic Areas (Sidney G. -J. Wong et al., 2023)

{{<citation>}}

Sidney G. -J. Wong, Jonathan Dunn, Benjamin Adams. (2023)  
**Comparing Measures of Linguistic Diversity Across Social Media Language Data and Census Data at Subnational Geographic Areas**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2308.10452v1)  

---


**ABSTRACT**  
This paper describes a preliminary study on the comparative linguistic ecology of online spaces (i.e., social media language data) and real-world spaces in Aotearoa New Zealand (i.e., subnational administrative areas). We compare measures of linguistic diversity between these different spaces and discuss how social media users align with real-world populations. The results from the current study suggests that there is potential to use online social media language data to observe spatial and temporal changes in linguistic diversity at subnational geographic areas; however, further work is required to understand how well social media represents real-world behaviour.

{{</citation>}}


### (108/125) Dynamic Strategy Chain: Dynamic Zero-Shot CoT for Long Mental Health Support Generation (Qi Chen et al., 2023)

{{<citation>}}

Qi Chen, Dexi Liu. (2023)  
**Dynamic Strategy Chain: Dynamic Zero-Shot CoT for Long Mental Health Support Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: GPT, Language Model, NLP, Text Generation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.10444v1)  

---


**ABSTRACT**  
Long counseling Text Generation for Mental health support (LTGM), an innovative and challenging task, aims to provide help-seekers with mental health support through a comprehensive and more acceptable response. The combination of chain-of-thought (CoT) prompting and Large Language Models (LLMs) is employed and get the SOTA performance on various NLP tasks, especially on text generation tasks. Zero-shot CoT prompting is one of the most common methods in CoT prompting. However, in the LTGM task, Zero-shot CoT prompting can not simulate a counselor or provide personalized strategies without effective mental health counseling strategy prompts. To tackle this challenge, we propose a zero-shot Dynamic Strategy Chain (DSC) prompting method. Firstly, we utilize GPT2 to learn the responses written by mental health counselors and dynamically generate mental health counseling strategies tailored to the help-seekers' needs. Secondly, the Zero-shot DSC prompting is constructed according to mental health counseling strategies and the help-seekers' post. Finally, the Zero-shot DSC prompting is employed to guide LLMs in generating more human-like responses for the help-seekers. Both automatic and manual evaluations demonstrate that Zero-shot DSC prompting can deliver more human-like responses than CoT prompting methods on LTGM tasks.

{{</citation>}}


### (109/125) Large Language Models on Wikipedia-Style Survey Generation: an Evaluation in NLP Concepts (Fan Gao et al., 2023)

{{<citation>}}

Fan Gao, Hang Jiang, Moritz Blum, Jinghui Lu, Yuang Jiang, Irene Li. (2023)  
**Large Language Models on Wikipedia-Style Survey Generation: an Evaluation in NLP Concepts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.10410v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have achieved significant success across various natural language processing (NLP) tasks, encompassing question-answering, summarization, and machine translation, among others. While LLMs excel in general tasks, their efficacy in domain-specific applications remains under exploration. Additionally, LLM-generated text sometimes exhibits issues like hallucination and disinformation. In this study, we assess LLMs' capability of producing concise survey articles within the computer science-NLP domain, focusing on 20 chosen topics. Automated evaluations indicate that GPT-4 outperforms GPT-3.5 when benchmarked against the ground truth. Furthermore, four human evaluators provide insights from six perspectives across four model configurations. Through case studies, we demonstrate that while GPT often yields commendable results, there are instances of shortcomings, such as incomplete information and the exhibition of lapses in factual accuracy.

{{</citation>}}


### (110/125) FairBench: A Four-Stage Automatic Framework for Detecting Stereotypes and Biases in Large Language Models (Yanhong Bai et al., 2023)

{{<citation>}}

Yanhong Bai, Jiabao Zhao, Jinxin Shi, Tingjiang Wei, Xingjiao Wu, Liang He. (2023)  
**FairBench: A Four-Stage Automatic Framework for Detecting Stereotypes and Biases in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10397v1)  

---


**ABSTRACT**  
Detecting stereotypes and biases in Large Language Models (LLMs) can enhance fairness and reduce adverse impacts on individuals or groups when these LLMs are applied. However, the majority of existing methods focus on measuring the model's preference towards sentences containing biases and stereotypes within datasets, which lacks interpretability and cannot detect implicit biases and stereotypes in the real world. To address this gap, this paper introduces a four-stage framework to directly evaluate stereotypes and biases in the generated content of LLMs, including direct inquiry testing, serial or adapted story testing, implicit association testing, and unknown situation testing. Additionally, the paper proposes multi-dimensional evaluation metrics and explainable zero-shot prompts for automated evaluation. Using the education sector as a case study, we constructed the Edu-FairBench based on the four-stage framework, which encompasses 12,632 open-ended questions covering nine sensitive factors and 26 educational scenarios. Experimental results reveal varying degrees of stereotypes and biases in five LLMs evaluated on Edu-FairBench. Moreover, the results of our proposed automated evaluation method have shown a high correlation with human annotations.

{{</citation>}}


## cs.GT (1)



### (111/125) Election Manipulation in Social Networks with Single-Peaked Agents (Vincenzo Auletta et al., 2023)

{{<citation>}}

Vincenzo Auletta, Francesco Carbone, Diodato Ferraioli. (2023)  
**Election Manipulation in Social Networks with Single-Peaked Agents**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs-MA, cs-SI, cs.GT  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.10845v1)  

---


**ABSTRACT**  
Several elections run in the last years have been characterized by attempts to manipulate the result of the election through the diffusion of fake or malicious news over social networks. This problem has been recognized as a critical issue for the robustness of our democracy. Analyzing and understanding how such manipulations may occur is crucial to the design of effective countermeasures to these practices.   Many studies have observed that, in general, to design an optimal manipulation is usually a computationally hard task. Nevertheless, literature on bribery in voting and election manipulation has frequently observed that most hardness results melt down when one focuses on the setting of (nearly) single-peaked agents, i.e., when each voter has a preferred candidate (usually, the one closer to her own belief) and preferences of remaining candidates are inversely proportional to the distance between the candidate position and the voter's belief. Unfortunately, no such analysis has been done for election manipulations run in social networks.   In this work, we try to close this gap: specifically, we consider a setting for election manipulation that naturally raises (nearly) single-peaked preferences, and we evaluate the complexity of election manipulation problem in this setting: while most of the hardness and approximation results still hold, we will show that single-peaked preferences allow to design simple, efficient and effective heuristics for election manipulation.

{{</citation>}}


## cs.IR (4)



### (112/125) Leveraging Large Language Models for Pre-trained Recommender Systems (Zhixuan Chu et al., 2023)

{{<citation>}}

Zhixuan Chu, Hongyan Hao, Xin Ouyang, Simeng Wang, Yan Wang, Yue Shen, Jinjie Gu, Qing Cui, Longfei Li, Siqiao Xue, James Y Zhang, Sheng Li. (2023)  
**Leveraging Large Language Models for Pre-trained Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10837v1)  

---


**ABSTRACT**  
Recent advancements in recommendation systems have shifted towards more comprehensive and personalized recommendations by utilizing large language models (LLM). However, effectively integrating LLM's commonsense knowledge and reasoning abilities into recommendation systems remains a challenging problem. In this paper, we propose RecSysLLM, a novel pre-trained recommendation model based on LLMs. RecSysLLM retains LLM reasoning and knowledge while integrating recommendation domain knowledge through unique designs of data, training, and inference. This allows RecSysLLM to leverage LLMs' capabilities for recommendation tasks in an efficient, unified framework. We demonstrate the effectiveness of RecSysLLM on benchmarks and real-world scenarios. RecSysLLM provides a promising approach to developing unified recommendation systems by fully exploiting the power of pre-trained language models.

{{</citation>}}


### (113/125) Enhancing Recommender Systems with Large Language Model Reasoning Graphs (Yan Wang et al., 2023)

{{<citation>}}

Yan Wang, Zhixuan Chu, Xin Ouyang, Simeng Wang, Hongyan Hao, Yue Shen, Jinjie Gu, Siqiao Xue, James Y Zhang, Qing Cui, Longfei Li, Jun Zhou, Sheng Li. (2023)  
**Enhancing Recommender Systems with Large Language Model Reasoning Graphs**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.10835v1)  

---


**ABSTRACT**  
Recommendation systems aim to provide users with relevant suggestions, but often lack interpretability and fail to capture higher-level semantic relationships between user behaviors and profiles. In this paper, we propose a novel approach that leverages large language models (LLMs) to construct personalized reasoning graphs. These graphs link a user's profile and behavioral sequences through causal and logical inferences, representing the user's interests in an interpretable way. Our approach, LLM reasoning graphs (LLMRG), has four components: chained graph reasoning, divergent extension, self-verification and scoring, and knowledge base self-improvement. The resulting reasoning graph is encoded using graph neural networks, which serves as additional input to improve conventional recommender systems, without requiring extra user or item information. Our approach demonstrates how LLMs can enable more logical and interpretable recommender systems through personalized reasoning graphs. LLMRG allows recommendations to benefit from both engineered recommendation systems and LLM-derived reasoning graphs. We demonstrate the effectiveness of LLMRG on benchmarks and real-world scenarios in enhancing base recommendation models.

{{</citation>}}


### (114/125) Contrastive Graph Prompt-tuning for Cross-domain Recommendation (Zixuan Yi et al., 2023)

{{<citation>}}

Zixuan Yi, Iadh Ounis, Craig Macdonald. (2023)  
**Contrastive Graph Prompt-tuning for Cross-domain Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Amazon, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.10685v1)  

---


**ABSTRACT**  
Recommender systems are frequently challenged by the data sparsity problem. One approach to mitigate this issue is through cross-domain recommendation techniques. In a cross-domain context, sharing knowledge between domains can enhance the effectiveness in the target domain. Recent cross-domain methods have employed a pre-training approach, but we argue that these methods often result in suboptimal fine-tuning, especially with large neural models. Modern language models utilize prompts for efficient model tuning. Such prompts act as a tunable latent vector, allowing for the freezing of the main model parameters. In our research, we introduce the Personalised Graph Prompt-based Recommendation (PGPRec) framework. This leverages the advantages of prompt-tuning. Within this framework, we formulate personalized graph prompts item-wise, rooted in items that a user has previously engaged with. Specifically, we employ Contrastive Learning (CL) to produce pre-trained embeddings that offer greater generalizability in the pre-training phase, ensuring robust training during the tuning phase. Our evaluation of PGPRec in cross-domain scenarios involves comprehensive testing with the top-k recommendation tasks and a cold-start analysis. Our empirical findings, based on four Amazon Review datasets, reveal that the PGPRec framework can decrease the tuned parameters by as much as 74%, maintaining competitive performance. Remarkably, there's an 11.41% enhancement in performance against the leading baseline in cold-start situations.

{{</citation>}}


### (115/125) Evaluating Temporal Persistence Using Replicability Measures (Jüri Keller et al., 2023)

{{<citation>}}

Jüri Keller, Timo Breuer, Philipp Schaer. (2023)  
**Evaluating Temporal Persistence Using Replicability Measures**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: BERT, Information Retrieval, T5  
[Paper Link](http://arxiv.org/abs/2308.10549v1)  

---


**ABSTRACT**  
In real-world Information Retrieval (IR) experiments, the Evaluation Environment (EE) is exposed to constant change. Documents are added, removed, or updated, and the information need and the search behavior of users is evolving. Simultaneously, IR systems are expected to retain a consistent quality. The LongEval Lab seeks to investigate the longitudinal persistence of IR systems, and in this work, we describe our participation. We submitted runs of five advanced retrieval systems, namely a Reciprocal Rank Fusion (RRF) approach, ColBERT, monoT5, Doc2Query, and E5, to both sub-tasks. Further, we cast the longitudinal evaluation as a replicability study to better understand the temporal change observed. As a result, we quantify the persistence of the submitted runs and see great potential in this evaluation method.

{{</citation>}}


## cs.HC (1)



### (116/125) Artificial intelligence is ineffective and potentially harmful for fact checking (Matthew R. DeVerna et al., 2023)

{{<citation>}}

Matthew R. DeVerna, Harry Yaojun Yan, Kai-Cheng Yang, Filippo Menczer. (2023)  
**Artificial intelligence is ineffective and potentially harmful for fact checking**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10800v1)  

---


**ABSTRACT**  
Fact checking can be an effective strategy against misinformation, but its implementation at scale is impeded by the overwhelming volume of information online. Recent artificial intelligence (AI) language models have shown impressive ability in fact-checking tasks, but how humans interact with fact-checking information provided by these models is unclear. Here we investigate the impact of fact checks generated by a popular AI model on belief in, and sharing intent of, political news in a preregistered randomized control experiment. Although the AI performs reasonably well in debunking false headlines, we find that it does not significantly affect participants' ability to discern headline accuracy or share accurate news. However, the AI fact-checker is harmful in specific cases: it decreases beliefs in true headlines that it mislabels as false and increases beliefs for false headlines that it is unsure about. On the positive side, the AI increases sharing intents for correctly labeled true headlines. When participants are given the option to view AI fact checks and choose to do so, they are significantly more likely to share both true and false news but only more likely to believe false news. Our findings highlight an important source of potential harm stemming from AI applications and underscore the critical need for policies to prevent or mitigate such unintended consequences.

{{</citation>}}


## quant-ph (1)



### (117/125) A Block-Ring connected Topology of Parameterized Quantum Circuits (Wenjie Liu et al., 2023)

{{<citation>}}

Wenjie Liu, Qingshan Wu. (2023)  
**A Block-Ring connected Topology of Parameterized Quantum Circuits**  

---
Primary Category: quant-ph  
Categories: cs-AI, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.10791v1)  

---


**ABSTRACT**  
It is essential to select efficient topology of parameterized quantum circuits (PQCs) in variational quantum algorithms (VQAs). However, there are problems in current circuits, i.e. optimization difficulties caused by too many parameters or performance is hard to guarantee. How to reduce the number of parameters (number of single-qubit rotation gates and 2-qubit gates) in PQCs without reducing the performance has become a new challenge. To solve this problem, we propose a novel topology, called Block-Ring (BR) topology, to construct the PQCs. This topology allocate all qubits to several blocks, all-to-all mode is adopt inside each block and ring mode is applied to connect different blocks. Compared with the pure all-to-all topology circuits which own the best power, BR topology have similar performance and the number of parameters and 2-qubit gate reduced from 0(n^2) to 0(mn) , m is a hyperparameter set by ourselves. Besides, we compared BR topology with other topology circuits in terms of expressibility and entangling capability. Considering the effects of different 2-qubit gates on circuits, we also make a distinction between controlled X-rotation gates and controlled Z-rotation gates. Finally, the 1- and 2-layer configurations of PQCs are taken into consideration as well, which shows the BR's performance improvement in the condition of multilayer circuits.

{{</citation>}}


## q-bio.BM (1)



### (118/125) Artificial intelligence-driven antimicrobial peptide discovery (Paulina Szymczak et al., 2023)

{{<citation>}}

Paulina Szymczak, Ewa Szczurek. (2023)  
**Artificial intelligence-driven antimicrobial peptide discovery**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10921v1)  

---


**ABSTRACT**  
Antimicrobial peptides (AMPs) emerge as promising agents against antimicrobial resistance, providing an alternative to conventional antibiotics. Artificial intelligence (AI) revolutionized AMP discovery through both discrimination and generation approaches. The discriminators aid the identification of promising candidates by predicting key peptide properties such as activity and toxicity, while the generators learn the distribution over peptides and enable sampling novel AMP candidates, either de novo, or as analogues of a prototype peptide. Moreover, the controlled generation of AMPs with desired properties is achieved by discriminator-guided filtering, positive-only learning, latent space sampling, as well as conditional and optimized generation. Here we review recent achievements in AI-driven AMP discovery, highlighting the most exciting directions.

{{</citation>}}


## cs.LO (1)



### (119/125) Normative Conditional Reasoning as a Fragment of HOL (Xavier Parent et al., 2023)

{{<citation>}}

Xavier Parent, Christoph Benzmüller. (2023)  
**Normative Conditional Reasoning as a Fragment of HOL**  

---
Primary Category: cs.LO  
Categories: 03B60, 03B15, 68T27, 68T30, 68T15, I-2-3; I-2-4; I-2-0; F-4, cs-AI, cs-LO, cs-SC, cs.LO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.10686v2)  

---


**ABSTRACT**  
We report some results regarding the mechanization of normative (preference-based) conditional reasoning. Our focus is on Aqvist's system E for conditional obligation (and its extensions). Our mechanization is achieved via a shallow semantical embedding in Isabelle/HOL. We consider two possible uses of the framework. The first one is as a tool for meta-reasoning about the considered logic. We employ it for the automated verification of deontic correspondences (broadly conceived) and related matters, analogous to what has been previously achieved for the modal logic cube. The second use is as a tool for assessing ethical arguments. We provide a computer encoding of a well-known paradox in population ethics, Parfit's repugnant conclusion. Whether the presented encoding increases or decreases the attractiveness and persuasiveness of the repugnant conclusion is a question we would like to pass on to philosophy and ethics.

{{</citation>}}


## cs.PF (1)



### (120/125) Algebraic Reasoning About Timeliness (Seyed Hossein Haeri et al., 2023)

{{<citation>}}

Seyed Hossein Haeri, Peter W. Thompson, Peter Van Roy, Magne Haveraaen, Neil J. Davies, Mikhail Barash, Kevin Hammond, James Chapman. (2023)  
**Algebraic Reasoning About Timeliness**  

---
Primary Category: cs.PF  
Categories: B-8-2; C-4; D-2-4; D-2-8; F-3-2; F-3-1; F-4-1; F-4-3; I-1-1, cs-DC, cs-PF, cs.PF  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.10654v1)  

---


**ABSTRACT**  
Designing distributed systems to have predictable performance under high load is difficult because of resource exhaustion, non-linearity, and stochastic behaviour. Timeliness, i.e., delivering results within defined time bounds, is a central aspect of predictable performance. In this paper, we focus on timeliness using the DELTA-Q Systems Development paradigm (DELTA-QSD, developed by PNSol), which computes timeliness by modelling systems observationally using so-called outcome expressions. An outcome expression is a compositional definition of a system's observed behaviour in terms of its basic operations. Given the behaviour of the basic operations, DELTA-QSD efficiently computes the stochastic behaviour of the whole system including its timeliness.   This paper formally proves useful algebraic properties of outcome expressions w.r.t. timeliness. We prove the different algebraic structures the set of outcome expressions form with the different DELTA-QSD operators and demonstrate why those operators do not form richer structures. We prove or disprove the set of all possible distributivity results on outcome expressions. On our way for disproving 8 of those distributivity results, we develop a technique called properisation, which gives rise to the first body of maths for improper random variables. Finally, we also prove 14 equivalences that have been used in the past in the practice of DELTA-QSD.   An immediate benefit is rewrite rules that can be used for design exploration under established timeliness equivalence. This work is part of an ongoing project to disseminate and build tool support for DELTA-QSD. The ability to rewrite outcome expressions is essential for efficient tool support.

{{</citation>}}


## eess.AS (1)



### (121/125) Implicit Self-supervised Language Representation for Spoken Language Diarization (Jagabandhu Mishra et al., 2023)

{{<citation>}}

Jagabandhu Mishra, S. R. Mahadeva Prasanna. (2023)  
**Implicit Self-supervised Language Representation for Spoken Language Diarization**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2308.10470v1)  

---


**ABSTRACT**  
In a code-switched (CS) scenario, the use of spoken language diarization (LD) as a pre-possessing system is essential. Further, the use of implicit frameworks is preferable over the explicit framework, as it can be easily adapted to deal with low/zero resource languages. Inspired by speaker diarization (SD) literature, three frameworks based on (1) fixed segmentation, (2) change point-based segmentation and (3) E2E are proposed to perform LD. The initial exploration with synthetic TTSF-LD dataset shows, using x-vector as implicit language representation with appropriate analysis window length ($N$) can able to achieve at per performance with explicit LD. The best implicit LD performance of $6.38$ in terms of Jaccard error rate (JER) is achieved by using the E2E framework. However, considering the E2E framework the performance of implicit LD degrades to $60.4$ while using with practical Microsoft CS (MSCS) dataset. The difference in performance is mostly due to the distributional difference between the monolingual segment duration of secondary language in the MSCS and TTSF-LD datasets. Moreover, to avoid segment smoothing, the smaller duration of the monolingual segment suggests the use of a small value of $N$. At the same time with small $N$, the x-vector representation is unable to capture the required language discrimination due to the acoustic similarity, as the same speaker is speaking both languages. Therefore, to resolve the issue a self-supervised implicit language representation is proposed in this study. In comparison with the x-vector representation, the proposed representation provides a relative improvement of $63.9\%$ and achieved a JER of $21.8$ using the E2E framework.

{{</citation>}}


## q-bio.QM (1)



### (122/125) PACS: Prediction and analysis of cancer subtypes from multi-omics data based on a multi-head attention mechanism model (Liangrui Pan et al., 2023)

{{<citation>}}

Liangrui Pan, Dazheng Liu, Zhichao Feng, Wenjuan Liu, Shaoliang Peng. (2023)  
**PACS: Prediction and analysis of cancer subtypes from multi-omics data based on a multi-head attention mechanism model**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, cs-MM, q-bio-QM, q-bio.QM  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.10917v1)  

---


**ABSTRACT**  
Due to the high heterogeneity and clinical characteristics of cancer, there are significant differences in multi-omic data and clinical characteristics among different cancer subtypes. Therefore, accurate classification of cancer subtypes can help doctors choose the most appropriate treatment options, improve treatment outcomes, and provide more accurate patient survival predictions. In this study, we propose a supervised multi-head attention mechanism model (SMA) to classify cancer subtypes successfully. The attention mechanism and feature sharing module of the SMA model can successfully learn the global and local feature information of multi-omics data. Second, it enriches the parameters of the model by deeply fusing multi-head attention encoders from Siamese through the fusion module. Validated by extensive experiments, the SMA model achieves the highest accuracy, F1 macroscopic, F1 weighted, and accurate classification of cancer subtypes in simulated, single-cell, and cancer multiomics datasets compared to AE, CNN, and GNN-based models. Therefore, we contribute to future research on multiomics data using our attention-based approach.

{{</citation>}}


## cs.SI (1)



### (123/125) DySuse: Susceptibility Estimation in Dynamic Social Networks (Yingdan Shi et al., 2023)

{{<citation>}}

Yingdan Shi, Jingya Zhou, Congcong Zhang. (2023)  
**DySuse: Susceptibility Estimation in Dynamic Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.10442v1)  

---


**ABSTRACT**  
Influence estimation aims to predict the total influence spread in social networks and has received surged attention in recent years. Most current studies focus on estimating the total number of influenced users in a social network, and neglect susceptibility estimation that aims to predict the probability of each user being influenced from the individual perspective. As a more fine-grained estimation task, susceptibility estimation is full of attractiveness and practical value. Based on the significance of susceptibility estimation and dynamic properties of social networks, we propose a task, called susceptibility estimation in dynamic social networks, which is even more realistic and valuable in real-world applications. Susceptibility estimation in dynamic networks has yet to be explored so far and is computationally intractable to naively adopt Monte Carlo simulation to obtain the results. To this end, we propose a novel end-to-end framework DySuse based on dynamic graph embedding technology. Specifically, we leverage a structural feature module to independently capture the structural information of influence diffusion on each single graph snapshot. Besides, {we propose the progressive mechanism according to the property of influence diffusion,} to couple the structural and temporal information during diffusion tightly. Moreover, a self-attention block {is designed to} further capture temporal dependency by flexibly weighting historical timestamps. Experimental results show that our framework is superior to the existing dynamic graph embedding models and has satisfactory prediction performance in multiple influence diffusion models.

{{</citation>}}


## stat.ML (1)



### (124/125) Approximately Equivariant Graph Networks (Ningyuan Huang et al., 2023)

{{<citation>}}

Ningyuan Huang, Ron Levie, Soledad Villar. (2023)  
**Approximately Equivariant Graph Networks**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.10436v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are commonly described as being permutation equivariant with respect to node relabeling in the graph. This symmetry of GNNs is often compared to the translation equivariance symmetry of Euclidean convolution neural networks (CNNs). However, these two symmetries are fundamentally different: The translation equivariance of CNNs corresponds to symmetries of the fixed domain acting on the image signal (sometimes known as active symmetries), whereas in GNNs any permutation acts on both the graph signals and the graph domain (sometimes described as passive symmetries). In this work, we focus on the active symmetries of GNNs, by considering a learning setting where signals are supported on a fixed graph. In this case, the natural symmetries of GNNs are the automorphisms of the graph. Since real-world graphs tend to be asymmetric, we relax the notion of symmetries by formalizing approximate symmetries via graph coarsening. We present a bias-variance formula that quantifies the tradeoff between the loss in expressivity and the gain in the regularity of the learned estimator, depending on the chosen symmetry group. To illustrate our approach, we conduct extensive experiments on image inpainting, traffic flow prediction, and human pose estimation with different choices of symmetries. We show theoretically and empirically that the best generalization performance can be achieved by choosing a suitably larger group than the graph automorphism group, but smaller than the full permutation group.

{{</citation>}}


## cs.MA (1)



### (125/125) GPT-in-the-Loop: Adaptive Decision-Making for Multiagent Systems (Nathalia Nascimento et al., 2023)

{{<citation>}}

Nathalia Nascimento, Paulo Alencar, Donald Cowan. (2023)  
**GPT-in-the-Loop: Adaptive Decision-Making for Multiagent Systems**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs-NE, cs-SE, cs.MA  
Keywords: GPT, GPT-4, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10435v1)  

---


**ABSTRACT**  
This paper introduces the "GPT-in-the-loop" approach, a novel method combining the advanced reasoning capabilities of Large Language Models (LLMs) like Generative Pre-trained Transformers (GPT) with multiagent (MAS) systems. Venturing beyond traditional adaptive approaches that generally require long training processes, our framework employs GPT-4 for enhanced problem-solving and explanation skills. Our experimental backdrop is the smart streetlight Internet of Things (IoT) application. Here, agents use sensors, actuators, and neural networks to create an energy-efficient lighting system. By integrating GPT-4, these agents achieve superior decision-making and adaptability without the need for extensive training. We compare this approach with both traditional neuroevolutionary methods and solutions provided by software engineers, underlining the potential of GPT-driven multiagent systems in IoT. Structurally, the paper outlines the incorporation of GPT into the agent-driven Framework for the Internet of Things (FIoT), introduces our proposed GPT-in-the-loop approach, presents comparative results in the IoT context, and concludes with insights and future directions.

{{</citation>}}
