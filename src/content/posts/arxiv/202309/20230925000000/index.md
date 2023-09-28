---
draft: false
title: "arXiv @ 2023.09.25"
date: 2023-09-25
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.25"
    identifier: arxiv_20230925
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [eess.AS (2)](#eessas-2)
- [cs.LG (10)](#cslg-10)
- [cs.CV (12)](#cscv-12)
- [stat.ML (2)](#statml-2)
- [cs.CL (19)](#cscl-19)
- [cs.AR (1)](#csar-1)
- [cs.CR (1)](#cscr-1)
- [eess.IV (2)](#eessiv-2)
- [cs.AI (1)](#csai-1)
- [cs.IR (2)](#csir-2)
- [cs.SD (1)](#cssd-1)
- [physics.geo-ph (1)](#physicsgeo-ph-1)
- [cs.NI (1)](#csni-1)
- [cs.SI (1)](#cssi-1)
- [cs.RO (2)](#csro-2)
- [cs.SE (1)](#csse-1)
- [cs.GT (1)](#csgt-1)
- [cs.IT (1)](#csit-1)

## eess.AS (2)



### (1/61) Attention Is All You Need For Blind Room Volume Estimation (Chunxi Wang et al., 2023)

{{<citation>}}

Chunxi Wang, Maoshen Jia, Meiran Li, Changchun Bao, Wenyu Jin. (2023)  
**Attention Is All You Need For Blind Room Volume Estimation**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.13504v1)  

---


**ABSTRACT**  
In recent years, dynamic parameterization of acoustic environments has raised increasing attention in the field of audio processing. One of the key parameters that characterize the local room acoustics in isolation from orientation and directivity of sources and receivers is the geometric room volume. Convolutional neural networks (CNNs) have been widely selected as the main models for conducting blind room acoustic parameter estimation, which aims to learn a direct mapping from audio spectrograms to corresponding labels. With the recent trend of self-attention mechanisms, this paper introduces a purely attention-based model to blindly estimate room volumes based on single-channel noisy speech signals. We demonstrate the feasibility of eliminating the reliance on CNN for this task and the proposed Transformer architecture takes Gammatone magnitude spectral coefficients and phase spectrograms as inputs. To enhance the model performance given the task-specific dataset, cross-modality transfer learning is also applied. Experimental results demonstrate that the proposed model outperforms traditional CNN models across a wide range of real-world acoustics spaces, especially with the help of the dedicated pretraining and data augmentation schemes.

{{</citation>}}


### (2/61) Contrastive Speaker Embedding With Sequential Disentanglement (Youzhi Tu et al., 2023)

{{<citation>}}

Youzhi Tu, Man-Wai Mak, Jen-Tzung Chien. (2023)  
**Contrastive Speaker Embedding With Sequential Disentanglement**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.13253v1)  

---


**ABSTRACT**  
Contrastive speaker embedding assumes that the contrast between the positive and negative pairs of speech segments is attributed to speaker identity only. However, this assumption is incorrect because speech signals contain not only speaker identity but also linguistic content. In this paper, we propose a contrastive learning framework with sequential disentanglement to remove linguistic content by incorporating a disentangled sequential variational autoencoder (DSVAE) into the conventional SimCLR framework. The DSVAE aims to disentangle speaker factors from content factors in an embedding space so that only the speaker factors are used for constructing a contrastive loss objective. Because content factors have been removed from the contrastive learning, the resulting speaker embeddings will be content-invariant. Experimental results on VoxCeleb1-test show that the proposed method consistently outperforms SimCLR. This suggests that applying sequential disentanglement is beneficial to learning speaker-discriminative embeddings.

{{</citation>}}


## cs.LG (10)



### (3/61) Enhancing Student Performance Prediction on Learnersourced Questions with SGNN-LLM Synergy (Lin Ni et al., 2023)

{{<citation>}}

Lin Ni, Sijie Wang, Zeyu Zhang, Xiaoxuan Li, Xianda Zheng, Paul Denny, Jiamou Liu. (2023)  
**Enhancing Student Performance Prediction on Learnersourced Questions with SGNN-LLM Synergy**  

---
Primary Category: cs.LG  
Categories: 97P80, cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13500v1)  

---


**ABSTRACT**  
As an emerging education strategy, learnersourcing offers the potential for personalized learning content creation, but also grapples with the challenge of predicting student performance due to inherent noise in student-generated data. While graph-based methods excel in capturing dense learner-question interactions, they falter in cold start scenarios, characterized by limited interactions, as seen when questions lack substantial learner responses. In response, we introduce an innovative strategy that synergizes the potential of integrating Signed Graph Neural Networks (SGNNs) and Large Language Model (LLM) embeddings. Our methodology employs a signed bipartite graph to comprehensively model student answers, complemented by a contrastive learning framework that enhances noise resilience. Furthermore, LLM's contribution lies in generating foundational question embeddings, proving especially advantageous in addressing cold start scenarios characterized by limited graph data interactions. Validation across five real-world datasets sourced from the PeerWise platform underscores our approach's effectiveness. Our method outperforms baselines, showcasing enhanced predictive accuracy and robustness.

{{</citation>}}


### (4/61) Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning (Berken Utku Demirel et al., 2023)

{{<citation>}}

Berken Utku Demirel, Christian Holz. (2023)  
**Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: Augmentation, Contrastive Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2309.13439v1)  

---


**ABSTRACT**  
The success of contrastive learning is well known to be dependent on data augmentation. Although the degree of data augmentations has been well controlled by utilizing pre-defined techniques in some domains like vision, time-series data augmentation is less explored and remains a challenging problem due to the complexity of the data generation mechanism, such as the intricate mechanism involved in the cardiovascular system. Moreover, there is no widely recognized and general time-series augmentation method that can be applied across different tasks. In this paper, we propose a novel data augmentation method for quasi-periodic time-series tasks that aims to connect intra-class samples together, and thereby find order in the latent space. Our method builds upon the well-known mixup technique by incorporating a novel approach that accounts for the periodic nature of non-stationary time-series. Also, by controlling the degree of chaos created by data augmentation, our method leads to improved feature representations and performance on downstream tasks. We evaluate our proposed method on three time-series tasks, including heart rate estimation, human activity recognition, and cardiovascular disease detection. Extensive experiments against state-of-the-art methods show that the proposed approach outperforms prior works on optimal data generation and known data augmentation techniques in the three tasks, reflecting the effectiveness of the presented method. Source code: https://github.com/eth-siplab/Finding_Order_in_Chaos

{{</citation>}}


### (5/61) MiliPoint: A Point Cloud Dataset for mmWave Radar (Han Cui et al., 2023)

{{<citation>}}

Han Cui, Shu Zhong, Jiacheng Wu, Zichao Shen, Naim Dahnoun, Yiren Zhao. (2023)  
**MiliPoint: A Point Cloud Dataset for mmWave Radar**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13425v1)  

---


**ABSTRACT**  
Millimetre-wave (mmWave) radar has emerged as an attractive and cost-effective alternative for human activity sensing compared to traditional camera-based systems. mmWave radars are also non-intrusive, providing better protection for user privacy. However, as a Radio Frequency (RF) based technology, mmWave radars rely on capturing reflected signals from objects, making them more prone to noise compared to cameras. This raises an intriguing question for the deep learning community: Can we develop more effective point set-based deep learning methods for such attractive sensors?   To answer this question, our work, termed MiliPoint, delves into this idea by providing a large-scale, open dataset for the community to explore how mmWave radars can be utilised for human activity recognition. Moreover, MiliPoint stands out as it is larger in size than existing datasets, has more diverse human actions represented, and encompasses all three key tasks in human activity recognition. We have also established a range of point-based deep neural networks such as DGCNN, PointNet++ and PointTransformer, on MiliPoint, which can serve to set the ground baseline for further development.

{{</citation>}}


### (6/61) Towards Attributions of Input Variables in a Coalition (Xinhao Zheng et al., 2023)

{{<citation>}}

Xinhao Zheng, Huiqi Deng, Quanshi Zhang. (2023)  
**Towards Attributions of Input Variables in a Coalition**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13411v1)  

---


**ABSTRACT**  
This paper aims to develop a new attribution method to explain the conflict between individual variables' attributions and their coalition's attribution from a fully new perspective. First, we find that the Shapley value can be reformulated as the allocation of Harsanyi interactions encoded by the AI model. Second, based the re-alloction of interactions, we extend the Shapley value to the attribution of coalitions. Third we ective. We derive the fundamental mechanism behind the conflict. This conflict come from the interaction containing partial variables in their coalition.

{{</citation>}}


### (7/61) Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment (Yutong Xia et al., 2023)

{{<citation>}}

Yutong Xia, Yuxuan Liang, Haomin Wen, Xu Liu, Kun Wang, Zhengyang Zhou, Roger Zimmermann. (2023)  
**Deciphering Spatio-Temporal Graph Forecasting: A Causal Lens and Treatment**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.13378v1)  

---


**ABSTRACT**  
Spatio-Temporal Graph (STG) forecasting is a fundamental task in many real-world applications. Spatio-Temporal Graph Neural Networks have emerged as the most popular method for STG forecasting, but they often struggle with temporal out-of-distribution (OoD) issues and dynamic spatial causation. In this paper, we propose a novel framework called CaST to tackle these two challenges via causal treatments. Concretely, leveraging a causal lens, we first build a structural causal model to decipher the data generation process of STGs. To handle the temporal OoD issue, we employ the back-door adjustment by a novel disentanglement block to separate invariant parts and temporal environments from input data. Moreover, we utilize the front-door adjustment and adopt the Hodge-Laplacian operator for edge-level convolution to model the ripple effect of causation. Experiments results on three real-world datasets demonstrate the effectiveness and practicality of CaST, which consistently outperforms existing methods with good interpretability.

{{</citation>}}


### (8/61) Limits of Actor-Critic Algorithms for Decision Tree Policies Learning in IBMDPs (Hecotr Kohler et al., 2023)

{{<citation>}}

Hecotr Kohler, Riad Akrour, Philippe Preux. (2023)  
**Limits of Actor-Critic Algorithms for Decision Tree Policies Learning in IBMDPs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.13365v1)  

---


**ABSTRACT**  
Interpretability of AI models allows for user safety checks to build trust in such AIs. In particular, Decision Trees (DTs) provide a global look at the learned model and transparently reveal which features of the input are critical for making a decision. However, interpretability is hindered if the DT is too large. To learn compact trees, a recent Reinforcement Learning (RL) framework has been proposed to explore the space of DTs using deep RL. This framework augments a decision problem (e.g. a supervised classification task) with additional actions that gather information about the features of an otherwise hidden input. By appropriately penalizing these actions, the agent learns to optimally trade-off size and performance of DTs. In practice, a reactive policy for a partially observable Markov decision process (MDP) needs to be learned, which is still an open problem. We show in this paper that deep RL can fail even on simple toy tasks of this class. However, when the underlying decision problem is a supervised classification task, we show that finding the optimal tree can be cast as a fully observable Markov decision problem and be solved efficiently, giving rise to a new family of algorithms for learning DTs that go beyond the classical greedy maximization ones.

{{</citation>}}


### (9/61) Predicting Temperature of Major Cities Using Machine Learning and Deep Learning (Wasiou Jaharabi et al., 2023)

{{<citation>}}

Wasiou Jaharabi, MD Ibrahim Al Hossain, Rownak Tahmid, Md. Zuhayer Islam, T. M. Saad Rayhan. (2023)  
**Predicting Temperature of Major Cities Using Machine Learning and Deep Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2309.13330v1)  

---


**ABSTRACT**  
Currently, the issue that concerns the world leaders most is climate change for its effect on agriculture, environment and economies of daily life. So, to combat this, temperature prediction with strong accuracy is vital. So far, the most effective widely used measure for such forecasting is Numerical weather prediction (NWP) which is a mathematical model that needs broad data from different applications to make predictions. This expensive, time and labor consuming work can be minimized through making such predictions using Machine learning algorithms. Using the database made by University of Dayton which consists the change of temperature in major cities we used the Time Series Analysis method where we use LSTM for the purpose of turning existing data into a tool for future prediction. LSTM takes the long-term data as well as any short-term exceptions or anomalies that may have occurred and calculates trend, seasonality and the stationarity of a data. By using models such as ARIMA, SARIMA, Prophet with the concept of RNN and LSTM we can, filter out any abnormalities, preprocess the data compare it with previous trends and make a prediction of future trends. Also, seasonality and stationarity help us analyze the reoccurrence or repeat over one year variable and removes the constrain of time in which the data was dependent so see the general changes that are predicted. By doing so we managed to make prediction of the temperature of different cities during any time in future based on available data and built a method of accurate prediction. This document contains our methodology for being able to make such predictions.

{{</citation>}}


### (10/61) An Interpretable Systematic Review of Machine Learning Models for Predictive Maintenance of Aircraft Engine (Abdullah Al Hasib et al., 2023)

{{<citation>}}

Abdullah Al Hasib, Ashikur Rahman, Mahpara Khabir, Md. Tanvir Rouf Shawon. (2023)  
**An Interpretable Systematic Review of Machine Learning Models for Predictive Maintenance of Aircraft Engine**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.13310v1)  

---


**ABSTRACT**  
This paper presents an interpretable review of various machine learning and deep learning models to predict the maintenance of aircraft engine to avoid any kind of disaster. One of the advantages of the strategy is that it can work with modest datasets. In this study, sensor data is utilized to predict aircraft engine failure within a predetermined number of cycles using LSTM, Bi-LSTM, RNN, Bi-RNN GRU, Random Forest, KNN, Naive Bayes, and Gradient Boosting. We explain how deep learning and machine learning can be used to generate predictions in predictive maintenance using a straightforward scenario with just one data source. We applied lime to the models to help us understand why machine learning models did not perform well than deep learning models. An extensive analysis of the model's behavior is presented for several test data to understand the black box scenario of the models. A lucrative accuracy of 97.8%, 97.14%, and 96.42% are achieved by GRU, Bi-LSTM, and LSTM respectively which denotes the capability of the models to predict maintenance at an early stage.

{{</citation>}}


### (11/61) Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks (Zhaohan Xi et al., 2023)

{{<citation>}}

Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Jinghui Chen, Fenglong Ma, Ting Wang. (2023)  
**Defending Pre-trained Language Models as Few-shot Learners against Backdoor Attacks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.13256v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) have demonstrated remarkable performance as few-shot learners. However, their security risks under such settings are largely unexplored. In this work, we conduct a pilot study showing that PLMs as few-shot learners are highly vulnerable to backdoor attacks while existing defenses are inadequate due to the unique challenges of few-shot scenarios. To address such challenges, we advocate MDP, a novel lightweight, pluggable, and effective defense for PLMs as few-shot learners. Specifically, MDP leverages the gap between the masking-sensitivity of poisoned and clean samples: with reference to the limited few-shot data as distributional anchors, it compares the representations of given samples under varying masking and identifies poisoned samples as ones with significant variations. We show analytically that MDP creates an interesting dilemma for the attacker to choose between attack effectiveness and detection evasiveness. The empirical evaluation using benchmark datasets and representative attacks validates the efficacy of MDP.

{{</citation>}}


### (12/61) COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs (Tiep Le et al., 2023)

{{<citation>}}

Tiep Le, Vasudev Lal, Phillip Howard. (2023)  
**COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.14356v1)  

---


**ABSTRACT**  
Counterfactual examples have proven to be valuable in the field of natural language processing (NLP) for both evaluating and improving the robustness of language models to spurious correlations in datasets. Despite their demonstrated utility for NLP, multimodal counterfactual examples have been relatively unexplored due to the difficulty of creating paired image-text data with minimal counterfactual changes. To address this challenge, we introduce a scalable framework for automatic generation of counterfactual examples using text-to-image diffusion models. We use our framework to create COCO-Counterfactuals, a multimodal counterfactual dataset of paired image and text captions based on the MS-COCO dataset. We validate the quality of COCO-Counterfactuals through human evaluations and show that existing multimodal models are challenged by our counterfactual image-text pairs. Additionally, we demonstrate the usefulness of COCO-Counterfactuals for improving out-of-domain generalization of multimodal vision-language models via training data augmentation.

{{</citation>}}


## cs.CV (12)



### (13/61) Portrait Stylization: Artistic Style Transfer with Auxiliary Networks for Human Face Stylization (Thiago Ambiel, 2023)

{{<citation>}}

Thiago Ambiel. (2023)  
**Portrait Stylization: Artistic Style Transfer with Auxiliary Networks for Human Face Stylization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.13492v1)  

---


**ABSTRACT**  
Today's image style transfer methods have difficulty retaining humans face individual features after the whole stylizing process. This occurs because the features like face geometry and people's expressions are not captured by the general-purpose image classifiers like the VGG-19 pre-trained models. This paper proposes the use of embeddings from an auxiliary pre-trained face recognition model to encourage the algorithm to propagate human face features from the content image to the final stylized result.

{{</citation>}}


### (14/61) HAVE-Net: Hallucinated Audio-Visual Embeddings for Few-Shot Classification with Unimodal Cues (Ankit Jha et al., 2023)

{{<citation>}}

Ankit Jha, Debabrata Pal, Mainak Singha, Naman Agarwal, Biplab Banerjee. (2023)  
**HAVE-Net: Hallucinated Audio-Visual Embeddings for Few-Shot Classification with Unimodal Cues**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.13470v1)  

---


**ABSTRACT**  
Recognition of remote sensing (RS) or aerial images is currently of great interest, and advancements in deep learning algorithms added flavor to it in recent years. Occlusion, intra-class variance, lighting, etc., might arise while training neural networks using unimodal RS visual input. Even though joint training of audio-visual modalities improves classification performance in a low-data regime, it has yet to be thoroughly investigated in the RS domain. Here, we aim to solve a novel problem where both the audio and visual modalities are present during the meta-training of a few-shot learning (FSL) classifier; however, one of the modalities might be missing during the meta-testing stage. This problem formulation is pertinent in the RS domain, given the difficulties in data acquisition or sensor malfunctioning. To mitigate, we propose a novel few-shot generative framework, Hallucinated Audio-Visual Embeddings-Network (HAVE-Net), to meta-train cross-modal features from limited unimodal data. Precisely, these hallucinated features are meta-learned from base classes and used for few-shot classification on novel classes during the inference phase. The experimental results on the benchmark ADVANCE and AudioSetZSL datasets show that our hallucinated modality augmentation strategy for few-shot classification outperforms the classifier performance trained with the real multimodal information at least by 0.8-2%.

{{</citation>}}


### (15/61) Beyond Grids: Exploring Elastic Input Sampling for Vision Transformers (Adam Pardyl et al., 2023)

{{<citation>}}

Adam Pardyl, Grzegorz Kurzejamski, Jan Olszewski, Tomasz Trzciński, Bartosz Zieliński. (2023)  
**Beyond Grids: Exploring Elastic Input Sampling for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.13353v1)  

---


**ABSTRACT**  
Vision transformers have excelled in various computer vision tasks but mostly rely on rigid input sampling using a fixed-size grid of patches. This limits their applicability in real-world problems, such as in the field of robotics and UAVs, where one can utilize higher input elasticity to boost model performance and efficiency. Our paper addresses this limitation by formalizing the concept of input elasticity for vision transformers and introducing an evaluation protocol, including dedicated metrics for measuring input elasticity. Moreover, we propose modifications to the transformer architecture and training regime, which increase its elasticity. Through extensive experimentation, we spotlight opportunities and challenges associated with input sampling strategies.

{{</citation>}}


### (16/61) FedDrive v2: an Analysis of the Impact of Label Skewness in Federated Semantic Segmentation for Autonomous Driving (Eros Fanì et al., 2023)

{{<citation>}}

Eros Fanì, Marco Ciccone, Barbara Caputo. (2023)  
**FedDrive v2: an Analysis of the Impact of Label Skewness in Federated Semantic Segmentation for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.13336v1)  

---


**ABSTRACT**  
We propose FedDrive v2, an extension of the Federated Learning benchmark for Semantic Segmentation in Autonomous Driving. While the first version aims at studying the effect of domain shift of the visual features across clients, in this work, we focus on the distribution skewness of the labels. We propose six new federated scenarios to investigate how label skewness affects the performance of segmentation models and compare it with the effect of domain shift. Finally, we study the impact of using the domain information during testing.

{{</citation>}}


### (17/61) Class Attendance System in Education with Deep Learning Method (Hüdaverdi Demir et al., 2023)

{{<citation>}}

Hüdaverdi Demir, Serkan Savaş. (2023)  
**Class Attendance System in Education with Deep Learning Method**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-HC, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13317v1)  

---


**ABSTRACT**  
With the advancing technology, the hardware gain of computers and the increase in the processing capacity of processors have facilitated the processing of instantaneous and real-time images. Face recognition processes are also studies in the field of image processing. Facial recognition processes are frequently used in security applications and commercial applications. Especially in the last 20 years, the high performances of artificial intelligence (AI) studies have contributed to the spread of these studies in many different fields. Education is one of them. The potential and advantages of using AI in education; can be grouped under three headings: student, teacher, and institution. One of the institutional studies may be the security of educational environments and the contribution of automation to education and training processes. From this point of view, deep learning methods, one of the sub-branches of AI, were used in this study. For object detection from images, a pioneering study has been designed and successfully implemented to keep records of students' entrance to the educational institution and to perform class attendance with images taken from the camera using image processing algorithms. The application of the study to real-life problems will be carried out in a school determined in the 2022-2023 academic year.

{{</citation>}}


### (18/61) Discwise Active Learning for LiDAR Semantic Segmentation (Ozan Unal et al., 2023)

{{<citation>}}

Ozan Unal, Dengxin Dai, Ali Tamer Unal, Luc Van Gool. (2023)  
**Discwise Active Learning for LiDAR Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.13276v1)  

---


**ABSTRACT**  
While LiDAR data acquisition is easy, labeling for semantic segmentation remains highly time consuming and must therefore be done selectively. Active learning (AL) provides a solution that can iteratively and intelligently label a dataset while retaining high performance and a low budget. In this work we explore AL for LiDAR semantic segmentation. As a human expert is a component of the pipeline, a practical framework must consider common labeling techniques such as sequential labeling that drastically improve annotation times. We therefore propose a discwise approach (DiAL), where in each iteration, we query the region a single frame covers on global coordinates, labeling all frames simultaneously. We then tackle the two major challenges that emerge with discwise AL. Firstly we devise a new acquisition function that takes 3D point density changes into consideration which arise due to location changes or ego-vehicle motion. Next we solve a mixed-integer linear program that provides a general solution to the selection of multiple frames while taking into consideration the possibilities of disc intersections. Finally we propose a semi-supervised learning approach to utilize all frames within our dataset and improve performance.

{{</citation>}}


### (19/61) Randomize to Generalize: Domain Randomization for Runway FOD Detection (Javaria Farooq et al., 2023)

{{<citation>}}

Javaria Farooq, Nayyer Aafaq, M Khizer Ali Khan, Ammar Saleem, M Ibraheem Siddiqui. (2023)  
**Randomize to Generalize: Domain Randomization for Runway FOD Detection**  

---
Primary Category: cs.CV  
Categories: I-4-9; I-5-4, cs-CV, cs.CV  
Keywords: Augmentation, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.13264v1)  

---


**ABSTRACT**  
Tiny Object Detection is challenging due to small size, low resolution, occlusion, background clutter, lighting conditions and small object-to-image ratio. Further, object detection methodologies often make underlying assumption that both training and testing data remain congruent. However, this presumption often leads to decline in performance when model is applied to out-of-domain(unseen) data. Techniques like synthetic image generation are employed to improve model performance by leveraging variations in input data. Such an approach typically presumes access to 3D-rendered datasets. In contrast, we propose a novel two-stage methodology Synthetic Randomized Image Augmentation (SRIA), carefully devised to enhance generalization capabilities of models encountering 2D datasets, particularly with lower resolution which is more practical in real-world scenarios. The first stage employs a weakly supervised technique to generate pixel-level segmentation masks. Subsequently, the second stage generates a batch-wise synthesis of artificial images, carefully designed with an array of diverse augmentations. The efficacy of proposed technique is illustrated on challenging foreign object debris (FOD) detection. We compare our results with several SOTA models including CenterNet, SSD, YOLOv3, YOLOv4, YOLOv5, and Outer Vit on a publicly available FOD-A dataset. We also construct an out-of-distribution test set encompassing 800 annotated images featuring a corpus of ten common categories. Notably, by harnessing merely 1.81% of objects from source training data and amalgamating with 29 runway background images, we generate 2227 synthetic images. Subsequent model retraining via transfer learning, utilizing enriched dataset generated by domain randomization, demonstrates significant improvement in detection accuracy. We report that detection accuracy improved from an initial 41% to 92% for OOD test set.

{{</citation>}}


### (20/61) Order-preserving Consistency Regularization for Domain Adaptation and Generalization (Mengmeng Jing et al., 2023)

{{<citation>}}

Mengmeng Jing, Xiantong Zhen, Jingjing Li, Cees Snoek. (2023)  
**Order-preserving Consistency Regularization for Domain Adaptation and Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2309.13258v1)  

---


**ABSTRACT**  
Deep learning models fail on cross-domain challenges if the model is oversensitive to domain-specific attributes, e.g., lightning, background, camera angle, etc. To alleviate this problem, data augmentation coupled with consistency regularization are commonly adopted to make the model less sensitive to domain-specific attributes. Consistency regularization enforces the model to output the same representation or prediction for two views of one image. These constraints, however, are either too strict or not order-preserving for the classification probabilities. In this work, we propose the Order-preserving Consistency Regularization (OCR) for cross-domain tasks. The order-preserving property for the prediction makes the model robust to task-irrelevant transformations. As a result, the model becomes less sensitive to the domain-specific attributes. The comprehensive experiments show that our method achieves clear advantages on five different cross-domain tasks.

{{</citation>}}


### (21/61) RBFormer: Improve Adversarial Robustness of Transformer by Robust Bias (Hao Cheng et al., 2023)

{{<citation>}}

Hao Cheng, Jinhao Duan, Hui Li, Lyutianyang Zhang, Jiahang Cao, Ping Wang, Jize Zhang, Kaidi Xu, Renjing Xu. (2023)  
**RBFormer: Improve Adversarial Robustness of Transformer by Robust Bias**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2309.13245v1)  

---


**ABSTRACT**  
Recently, there has been a surge of interest and attention in Transformer-based structures, such as Vision Transformer (ViT) and Vision Multilayer Perceptron (VMLP). Compared with the previous convolution-based structures, the Transformer-based structure under investigation showcases a comparable or superior performance under its distinctive attention-based input token mixer strategy. Introducing adversarial examples as a robustness consideration has had a profound and detrimental impact on the performance of well-established convolution-based structures. This inherent vulnerability to adversarial attacks has also been demonstrated in Transformer-based structures. In this paper, our emphasis lies on investigating the intrinsic robustness of the structure rather than introducing novel defense measures against adversarial attacks. To address the susceptibility to robustness issues, we employ a rational structure design approach to mitigate such vulnerabilities. Specifically, we enhance the adversarial robustness of the structure by increasing the proportion of high-frequency structural robust biases. As a result, we introduce a novel structure called Robust Bias Transformer-based Structure (RBFormer) that shows robust superiority compared to several existing baseline structures. Through a series of extensive experiments, RBFormer outperforms the original structures by a significant margin, achieving an impressive improvement of +16.12% and +5.04% across different evaluation criteria on CIFAR-10 and ImageNet-1k, respectively.

{{</citation>}}


### (22/61) UniHead: Unifying Multi-Perception for Detection Heads (Hantao Zhou et al., 2023)

{{<citation>}}

Hantao Zhou, Rui Yang, Yachao Zhang, Haoran Duan, Yawen Huang, Runze Hu, Xiu Li, Yefeng Zheng. (2023)  
**UniHead: Unifying Multi-Perception for Detection Heads**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13242v1)  

---


**ABSTRACT**  
The detection head constitutes a pivotal component within object detectors, tasked with executing both classification and localization functions. Regrettably, the commonly used parallel head often lacks omni perceptual capabilities, such as deformation perception, global perception and cross-task perception. Despite numerous methods attempt to enhance these abilities from a single aspect, achieving a comprehensive and unified solution remains a significant challenge. In response to this challenge, we have developed an innovative detection head, termed UniHead, to unify three perceptual abilities simultaneously. More precisely, our approach (1) introduces deformation perception, enabling the model to adaptively sample object features; (2) proposes a Dual-axial Aggregation Transformer (DAT) to adeptly model long-range dependencies, thereby achieving global perception; and (3) devises a Cross-task Interaction Transformer (CIT) that facilitates interaction between the classification and localization branches, thus aligning the two tasks. As a plug-and-play method, the proposed UniHead can be conveniently integrated with existing detectors. Extensive experiments on the COCO dataset demonstrate that our UniHead can bring significant improvements to many detectors. For instance, the UniHead can obtain +2.7 AP gains in RetinaNet, +2.9 AP gains in FreeAnchor, and +2.1 AP gains in GFL. The code will be publicly available. Code Url: https://github.com/zht8506/UniHead.

{{</citation>}}


### (23/61) Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation (Tao Pu et al., 2023)

{{<citation>}}

Tao Pu, Tianshui Chen, Hefeng Wu, Yongyi Lu, Liang Lin. (2023)  
**Spatial-Temporal Knowledge-Embedded Transformer for Video Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13237v1)  

---


**ABSTRACT**  
Video scene graph generation (VidSGG) aims to identify objects in visual scenes and infer their relationships for a given video. It requires not only a comprehensive understanding of each object scattered on the whole scene but also a deep dive into their temporal motions and interactions. Inherently, object pairs and their relationships enjoy spatial co-occurrence correlations within each image and temporal consistency/transition correlations across different images, which can serve as prior knowledge to facilitate VidSGG model learning and inference. In this work, we propose a spatial-temporal knowledge-embedded transformer (STKET) that incorporates the prior spatial-temporal knowledge into the multi-head cross-attention mechanism to learn more representative relationship representations. Specifically, we first learn spatial co-occurrence and temporal transition correlations in a statistical manner. Then, we design spatial and temporal knowledge-embedded layers that introduce the multi-head cross-attention mechanism to fully explore the interaction between visual representation and the knowledge to generate spatial- and temporal-embedded representations, respectively. Finally, we aggregate these representations for each subject-object pair to predict the final semantic labels and their relationships. Extensive experiments show that STKET outperforms current competing algorithms by a large margin, e.g., improving the mR@50 by 8.1%, 4.7%, and 2.1% on different settings over current algorithms.

{{</citation>}}


### (24/61) Real3D-AD: A Dataset of Point Cloud Anomaly Detection (Jiaqi Liu et al., 2023)

{{<citation>}}

Jiaqi Liu, Guoyang Xie, Ruitao Chen, Xinpeng Li, Jinbao Wang, Yong Liu, Chengjie Wang, Feng Zheng. (2023)  
**Real3D-AD: A Dataset of Point Cloud Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.13226v2)  

---


**ABSTRACT**  
High-precision point cloud anomaly detection is the gold standard for identifying the defects of advancing machining and precision manufacturing. Despite some methodological advances in this area, the scarcity of datasets and the lack of a systematic benchmark hinder its development. We introduce Real3D-AD, a challenging high-precision point cloud anomaly detection dataset, addressing the limitations in the field. With 1,254 high-resolution 3D items from forty thousand to millions of points for each item, Real3D-AD is the largest dataset for high-precision 3D industrial anomaly detection to date. Real3D-AD surpasses existing 3D anomaly detection datasets available regarding point cloud resolution (0.0010mm-0.0015mm), 360 degree coverage and perfect prototype. Additionally, we present a comprehensive benchmark for Real3D-AD, revealing the absence of baseline methods for high-precision point cloud anomaly detection. To address this, we propose Reg3D-AD, a registration-based 3D anomaly detection method incorporating a novel feature memory bank that preserves local and global representations. Extensive experiments on the Real3D-AD dataset highlight the effectiveness of Reg3D-AD. For reproducibility and accessibility, we provide the Real3D-AD dataset, benchmark source code, and Reg3D-AD on our website:https://github.com/M-3LAB/Real3D-AD.

{{</citation>}}


## stat.ML (2)



### (25/61) Enhancing Prediction and Analysis of UK Road Traffic Accident Severity Using AI: Integration of Machine Learning, Econometric Techniques, and Time Series Forecasting in Public Health Research (Md Abu Sufian et al., 2023)

{{<citation>}}

Md Abu Sufian, Jayasree Varadarajan. (2023)  
**Enhancing Prediction and Analysis of UK Road Traffic Accident Severity Using AI: Integration of Machine Learning, Econometric Techniques, and Time Series Forecasting in Public Health Research**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: AI, Time Series  
[Paper Link](http://arxiv.org/abs/2309.13483v1)  

---


**ABSTRACT**  
This research investigates road traffic accident severity in the UK, using a combination of machine learning, econometric, and statistical methods on historical data. We employed various techniques, including correlation analysis, regression models, GMM for error term issues, and time-series forecasting with VAR and ARIMA models. Our approach outperforms naive forecasting with an MASE of 0.800 and ME of -73.80. We also built a random forest classifier with 73% precision, 78% recall, and a 73% F1-score. Optimizing with H2O AutoML led to an XGBoost model with an RMSE of 0.176 and MAE of 0.087. Factor Analysis identified key variables, and we used SHAP for Explainable AI, highlighting influential factors like Driver_Home_Area_Type and Road_Type. Our study enhances understanding of accident severity and offers insights for evidence-based road safety policies.

{{</citation>}}


### (26/61) A Model-Agnostic Graph Neural Network for Integrating Local and Global Information (Wenzhuo Zhou et al., 2023)

{{<citation>}}

Wenzhuo Zhou, Annie Qu, Keiland W. Cooper, Norbert Fortin, Babak Shahbaba. (2023)  
**A Model-Agnostic Graph Neural Network for Integrating Local and Global Information**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.13459v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have achieved promising performance in a variety of graph-focused tasks. Despite their success, existing GNNs suffer from two significant limitations: a lack of interpretability in results due to their black-box nature, and an inability to learn representations of varying orders. To tackle these issues, we propose a novel Model-agnostic Graph Neural Network (MaGNet) framework, which is able to sequentially integrate information of various orders, extract knowledge from high-order neighbors, and provide meaningful and interpretable results by identifying influential compact graph structures. In particular, MaGNet consists of two components: an estimation model for the latent representation of complex relationships under graph topology, and an interpretation model that identifies influential nodes, edges, and important node features. Theoretically, we establish the generalization error bound for MaGNet via empirical Rademacher complexity, and showcase its power to represent layer-wise neighborhood mixing. We conduct comprehensive numerical studies using simulated data to demonstrate the superior performance of MaGNet in comparison to several state-of-the-art alternatives. Furthermore, we apply MaGNet to a real-world case study aimed at extracting task-critical information from brain activity data, thereby highlighting its effectiveness in advancing scientific research.

{{</citation>}}


## cs.CL (19)



### (27/61) Grounding Description-Driven Dialogue State Trackers with Knowledge-Seeking Turns (Alexandru Coca et al., 2023)

{{<citation>}}

Alexandru Coca, Bo-Hsiang Tseng, Jinghong Chen, Weizhe Lin, Weixuan Zhang, Tisha Anders, Bill Byrne. (2023)  
**Grounding Description-Driven Dialogue State Trackers with Knowledge-Seeking Turns**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.13448v1)  

---


**ABSTRACT**  
Schema-guided dialogue state trackers can generalise to new domains without further training, yet they are sensitive to the writing style of the schemata. Augmenting the training set with human or synthetic schema paraphrases improves the model robustness to these variations but can be either costly or difficult to control. We propose to circumvent these issues by grounding the state tracking model in knowledge-seeking turns collected from the dialogue corpus as well as the schema. Including these turns in prompts during finetuning and inference leads to marked improvements in model robustness, as demonstrated by large average joint goal accuracy and schema sensitivity improvements on SGD and SGD-X.

{{</citation>}}


### (28/61) Resolving References in Visually-Grounded Dialogue via Text Generation (Bram Willemsen et al., 2023)

{{<citation>}}

Bram Willemsen, Livia Qian, Gabriel Skantze. (2023)  
**Resolving References in Visually-Grounded Dialogue via Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Dialog, Dialogue, Text Generation  
[Paper Link](http://arxiv.org/abs/2309.13430v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) have shown to be effective at image retrieval based on simple text queries, but text-image retrieval based on conversational input remains a challenge. Consequently, if we want to use VLMs for reference resolution in visually-grounded dialogue, the discourse processing capabilities of these models need to be augmented. To address this issue, we propose fine-tuning a causal large language model (LLM) to generate definite descriptions that summarize coreferential information found in the linguistic context of references. We then use a pretrained VLM to identify referents based on the generated descriptions, zero-shot. We evaluate our approach on a manually annotated dataset of visually-grounded dialogues and achieve results that, on average, exceed the performance of the baselines we compare against. Furthermore, we find that using referent descriptions based on larger context windows has the potential to yield higher returns.

{{</citation>}}


### (29/61) A Chat About Boring Problems: Studying GPT-based text normalization (Yang Zhang et al., 2023)

{{<citation>}}

Yang Zhang, Travis M. Bartley, Mariana Graterol-Fuenmayor, Vitaly Lavrukhin, Evelina Bakhturina, Boris Ginsburg. (2023)  
**A Chat About Boring Problems: Studying GPT-based text normalization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13426v1)  

---


**ABSTRACT**  
Text normalization - the conversion of text from written to spoken form - is traditionally assumed to be an ill-formed task for language models. In this work, we argue otherwise. We empirically show the capacity of Large-Language Models (LLM) for text normalization in few-shot scenarios. Combining self-consistency reasoning with linguistic-informed prompt engineering, we find LLM based text normalization to achieve error rates around 40\% lower than top normalization systems. Further, upon error analysis, we note key limitations in the conventional design of text normalization tasks. We create a new taxonomy of text normalization errors and apply it to results from GPT-3.5-Turbo and GPT-4.0. Through this new framework, we can identify strengths and weaknesses of GPT-based TN, opening opportunities for future work.

{{</citation>}}


### (30/61) Exploring Large Language Models' Cognitive Moral Development through Defining Issues Test (Kumar Tanmay et al., 2023)

{{<citation>}}

Kumar Tanmay, Aditi Khandelwal, Utkarsh Agarwal, Monojit Choudhury. (2023)  
**Exploring Large Language Models' Cognitive Moral Development through Defining Issues Test**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13356v1)  

---


**ABSTRACT**  
The development of large language models has instilled widespread interest among the researchers to understand their inherent reasoning and problem-solving capabilities. Despite good amount of research going on to elucidate these capabilities, there is a still an appreciable gap in understanding moral development and judgments of these models. The current approaches of evaluating the ethical reasoning abilities of these models as a classification task pose numerous inaccuracies because of over-simplification. In this study, we built a psychological connection by bridging two disparate fields-human psychology and AI. We proposed an effective evaluation framework which can help to delineate the model's ethical reasoning ability in terms of moral consistency and Kohlberg's moral development stages with the help of Psychometric Assessment Tool-Defining Issues Test.

{{</citation>}}


### (31/61) Lexical Squad@Multimodal Hate Speech Event Detection 2023: Multimodal Hate Speech Detection using Fused Ensemble Approach (Mohammad Kashif et al., 2023)

{{<citation>}}

Mohammad Kashif, Mohammad Zohair, Saquib Ali. (2023)  
**Lexical Squad@Multimodal Hate Speech Event Detection 2023: Multimodal Hate Speech Detection using Fused Ensemble Approach**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Event Detection, Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2309.13354v1)  

---


**ABSTRACT**  
With a surge in the usage of social media postings to express opinions, emotions, and ideologies, there has been a significant shift towards the calibration of social media as a rapid medium of conveying viewpoints and outlooks over the globe. Concurrently, the emergence of a multitude of conflicts between two entities has given rise to a stream of social media content containing propaganda, hate speech, and inconsiderate views. Thus, the issue of monitoring social media postings is rising swiftly, attracting major attention from those willing to solve such problems. One such problem is Hate Speech detection. To mitigate this problem, we present our novel ensemble learning approach for detecting hate speech, by classifying text-embedded images into two labels, namely "Hate Speech" and "No Hate Speech". We have incorporated state-of-art models including InceptionV3, BERT, and XLNet. Our proposed ensemble model yielded promising results with 75.21 and 74.96 as accuracy and F-1 score (respectively). We also present an empirical evaluation of the text-embedded images to elaborate on how well the model was able to predict and classify. We release our codebase here (https://github.com/M0hammad-Kashif/MultiModalHateSpeech).

{{</citation>}}


### (32/61) My Science Tutor (MyST) -- A Large Corpus of Children's Conversational Speech (Sameer S. Pradhan et al., 2023)

{{<citation>}}

Sameer S. Pradhan, Ronald A. Cole, Wayne H. Ward. (2023)  
**My Science Tutor (MyST) -- A Large Corpus of Children's Conversational Speech**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13347v1)  

---


**ABSTRACT**  
This article describes the MyST corpus developed as part of the My Science Tutor project -- one of the largest collections of children's conversational speech comprising approximately 400 hours, spanning some 230K utterances across about 10.5K virtual tutor sessions by around 1.3K third, fourth and fifth grade students. 100K of all utterances have been transcribed thus far. The corpus is freely available (https://myst.cemantix.org) for non-commercial use using a creative commons license. It is also available for commercial use (https://boulderlearning.com/resources/myst-corpus/). To date, ten organizations have licensed the corpus for commercial use, and approximately 40 university and other not-for-profit research groups have downloaded the corpus. It is our hope that the corpus can be used to improve automatic speech recognition algorithms, build and evaluate conversational AI agents for education, and together help accelerate development of multimodal applications to improve children's excitement and learning about science, and help them learn remotely.

{{</citation>}}


### (33/61) BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models (Zican Dong et al., 2023)

{{<citation>}}

Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao, Ji-Rong Wen. (2023)  
**BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.13345v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved dramatic proficiency over NLP tasks with normal length. Recently, multiple studies have committed to extending the context length and enhancing the long text modeling capabilities of LLMs. To comprehensively evaluate the long context ability of LLMs, we propose BAMBOO, a multi-task long context benchmark. BAMBOO has been designed with four principles: comprehensive capacity evaluation, avoidance of data contamination, accurate automatic evaluation, and different length levels. It consists of 10 datasets from 5 different long text understanding tasks, i.e. question answering, hallucination detection, text sorting, language modeling, and code completion, to cover core capacities and various domains of LLMs. We conduct experiments with five long context models on BAMBOO and further discuss four key research questions of long text. We also qualitatively analyze current long context models and point out future directions for enhancing long text modeling capacities. We release our data, prompts, and code at https://github.com/RUCAIBox/BAMBOO.

{{</citation>}}


### (34/61) An In-depth Survey of Large Language Model-based Artificial Intelligence Agents (Pengyu Zhao et al., 2023)

{{<citation>}}

Pengyu Zhao, Zijian Jin, Ning Cheng. (2023)  
**An In-depth Survey of Large Language Model-based Artificial Intelligence Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.14365v1)  

---


**ABSTRACT**  
Due to the powerful capabilities demonstrated by large language model (LLM), there has been a recent surge in efforts to integrate them with AI agents to enhance their performance. In this paper, we have explored the core differences and characteristics between LLM-based AI agents and traditional AI agents. Specifically, we first compare the fundamental characteristics of these two types of agents, clarifying the significant advantages of LLM-based agents in handling natural language, knowledge storage, and reasoning capabilities. Subsequently, we conducted an in-depth analysis of the key components of AI agents, including planning, memory, and tool use. Particularly, for the crucial component of memory, this paper introduced an innovative classification scheme, not only departing from traditional classification methods but also providing a fresh perspective on the design of an AI agent's memory system. We firmly believe that in-depth research and understanding of these core components will lay a solid foundation for the future advancement of AI agent technology. At the end of the paper, we provide directional suggestions for further research in this field, with the hope of offering valuable insights to scholars and researchers in the field.

{{</citation>}}


### (35/61) LLMs as Counterfactual Explanation Modules: Can ChatGPT Explain Black-box Text Classifiers? (Amrita Bhattacharjee et al., 2023)

{{<citation>}}

Amrita Bhattacharjee, Raha Moraffah, Joshua Garland, Huan Liu. (2023)  
**LLMs as Counterfactual Explanation Modules: Can ChatGPT Explain Black-box Text Classifiers?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2309.13340v1)  

---


**ABSTRACT**  
Large language models (LLMs) are increasingly being used for tasks beyond text generation, including complex tasks such as data labeling, information extraction, etc. With the recent surge in research efforts to comprehend the full extent of LLM capabilities, in this work, we investigate the role of LLMs as counterfactual explanation modules, to explain decisions of black-box text classifiers. Inspired by causal thinking, we propose a pipeline for using LLMs to generate post-hoc, model-agnostic counterfactual explanations in a principled way via (i) leveraging the textual understanding capabilities of the LLM to identify and extract latent features, and (ii) leveraging the perturbation and generation capabilities of the same LLM to generate a counterfactual explanation by perturbing input features derived from the extracted latent features. We evaluate three variants of our framework, with varying degrees of specificity, on a suite of state-of-the-art LLMs, including ChatGPT and LLaMA 2. We evaluate the effectiveness and quality of the generated counterfactual explanations, over a variety of text classification benchmarks. Our results show varied performance of these models in different settings, with a full two-step feature extraction based variant outperforming others in most cases. Our pipeline can be used in automated explanation systems, potentially reducing human effort.

{{</citation>}}


### (36/61) Enhancing Zero-Shot Chain-of-Thought Reasoning in Large Language Models through Logic (Xufeng Zhao et al., 2023)

{{<citation>}}

Xufeng Zhao, Mengdi Li, Wenhao Lu, Cornelius Weber, Jae Hee Lee, Kun Chu, Stefan Wermter. (2023)  
**Enhancing Zero-Shot Chain-of-Thought Reasoning in Large Language Models through Logic**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SC, cs.CL  
Keywords: Language Model, Reasoning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.13339v1)  

---


**ABSTRACT**  
Recent advancements in large language models have showcased their remarkable generalizability across various domains. However, their reasoning abilities still have significant room for improvement, especially when confronted with scenarios requiring multi-step reasoning. Although large language models possess extensive knowledge, their behavior, particularly in terms of reasoning, often fails to effectively utilize this knowledge to establish a coherent thinking paradigm. Generative language models sometimes show hallucinations as their reasoning procedures are unconstrained by logical principles. Aiming to improve the zero-shot chain-of-thought reasoning ability of large language models, we propose Logical Chain-of-Thought (LogiCoT), a neurosymbolic framework that leverages principles from symbolic logic to verify and revise the reasoning processes accordingly. Experimental evaluations conducted on language tasks in diverse domains, including arithmetic, commonsense, symbolic, causal inference, and social problems, demonstrate the efficacy of the enhanced reasoning paradigm by logic.

{{</citation>}}


### (37/61) Diversifying Question Generation over Knowledge Base via External Natural Questions (Shasha Guo et al., 2023)

{{<citation>}}

Shasha Guo, Jing Zhang, Xirui Ke, Cuiping Li, Hong Chen. (2023)  
**Diversifying Question Generation over Knowledge Base via External Natural Questions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Question Generation  
[Paper Link](http://arxiv.org/abs/2309.14362v1)  

---


**ABSTRACT**  
Previous methods on knowledge base question generation (KBQG) primarily focus on enhancing the quality of a single generated question. Recognizing the remarkable paraphrasing ability of humans, we contend that diverse texts should convey the same semantics through varied expressions. The above insights make diversifying question generation an intriguing task, where the first challenge is evaluation metrics for diversity. Current metrics inadequately assess the above diversity since they calculate the ratio of unique n-grams in the generated question itself, which leans more towards measuring duplication rather than true diversity. Accordingly, we devise a new diversity evaluation metric, which measures the diversity among top-k generated questions for each instance while ensuring their relevance to the ground truth. Clearly, the second challenge is how to enhance diversifying question generation. To address this challenge, we introduce a dual model framework interwoven by two selection strategies to generate diverse questions leveraging external natural questions. The main idea of our dual framework is to extract more diverse expressions and integrate them into the generation model to enhance diversifying question generation. Extensive experiments on widely used benchmarks for KBQG demonstrate that our proposed approach generates highly diverse questions and improves the performance of question answering tasks.

{{</citation>}}


### (38/61) From Text to Source: Results in Detecting Large Language Model-Generated Content (Wissam Antoun et al., 2023)

{{<citation>}}

Wissam Antoun, Benoît Sagot, Djamé Seddah. (2023)  
**From Text to Source: Results in Detecting Large Language Model-Generated Content**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.13322v1)  

---


**ABSTRACT**  
The widespread use of Large Language Models (LLMs), celebrated for their ability to generate human-like text, has raised concerns about misinformation and ethical implications. Addressing these concerns necessitates the development of robust methods to detect and attribute text generated by LLMs. This paper investigates "Cross-Model Detection," evaluating whether a classifier trained to distinguish between source LLM-generated and human-written text can also detect text from a target LLM without further training. The study comprehensively explores various LLM sizes and families, and assesses the impact of conversational fine-tuning techniques on classifier generalization. The research also delves into Model Attribution, encompassing source model identification, model family classification, and model size classification. Our results reveal several key findings: a clear inverse relationship between classifier effectiveness and model size, with larger LLMs being more challenging to detect, especially when the classifier is trained on data from smaller models. Training on data from similarly sized LLMs can improve detection performance from larger models but may lead to decreased performance when dealing with smaller models. Additionally, model attribution experiments show promising results in identifying source models and model families, highlighting detectable signatures in LLM-generated text. Overall, our study contributes valuable insights into the interplay of model size, family, and training data in LLM detection and attribution.

{{</citation>}}


### (39/61) GlotScript: A Resource and Tool for Low Resource Writing System Identification (Amir Hossein Kargaran et al., 2023)

{{<citation>}}

Amir Hossein Kargaran, François Yvon, Hinrich Schütze. (2023)  
**GlotScript: A Resource and Tool for Low Resource Writing System Identification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2309.13320v1)  

---


**ABSTRACT**  
We present GlotScript, an open resource and tool for low resource writing system identification. GlotScript-R is a resource that provides the attested writing systems for more than 7,000 languages. It is compiled by aggregating information from existing writing system resources. GlotScript-T is a writing system identification tool that covers all 161 Unicode 15.0 scripts. For an input text, it returns its script distribution where scripts are identified by ISO 15924 codes. We also present two use cases for GlotScript. First, we demonstrate that GlotScript supports cleaning multilingual corpora such as mC4 and OSCAR. Second, we analyze the tokenization of a number of language models such as GPT-4 using GlotScript and provide insights on the coverage of low resource scripts and languages by each language model. We hope that GlotScript will become a useful resource for work on low resource languages in the NLP community. GlotScript-R and GlotScript-T are available at https://github.com/cisnlp/GlotScript.

{{</citation>}}


### (40/61) OATS: Opinion Aspect Target Sentiment Quadruple Extraction Dataset for Aspect-Based Sentiment Analysis (Siva Uday Sampreeth Chebolu et al., 2023)

{{<citation>}}

Siva Uday Sampreeth Chebolu, Franck Dernoncourt, Nedim Lipka, Thamar Solorio. (2023)  
**OATS: Opinion Aspect Target Sentiment Quadruple Extraction Dataset for Aspect-Based Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.13297v1)  

---


**ABSTRACT**  
Aspect-based sentiment Analysis (ABSA) delves into understanding sentiments specific to distinct elements within textual content. It aims to analyze user-generated reviews to determine a) the target entity being reviewed, b) the high-level aspect to which it belongs, c) the sentiment words used to express the opinion, and d) the sentiment expressed toward the targets and the aspects. While various benchmark datasets have fostered advancements in ABSA, they often come with domain limitations and data granularity challenges. Addressing these, we introduce the OATS dataset, which encompasses three fresh domains and consists of 20,000 sentence-level quadruples and 13,000 review-level tuples. Our initiative seeks to bridge specific observed gaps: the recurrent focus on familiar domains like restaurants and laptops, limited data for intricate quadruple extraction tasks, and an occasional oversight of the synergy between sentence and review-level sentiments. Moreover, to elucidate OATS's potential and shed light on various ABSA subtasks that OATS can solve, we conducted in-domain and cross-domain experiments, establishing initial baselines. We hope the OATS dataset augments current resources, paving the way for an encompassing exploration of ABSA.

{{</citation>}}


### (41/61) A Survey of Document-Level Information Extraction (Hanwen Zheng et al., 2023)

{{<citation>}}

Hanwen Zheng, Sijia Wang, Lifu Huang. (2023)  
**A Survey of Document-Level Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, NLP  
[Paper Link](http://arxiv.org/abs/2309.13249v1)  

---


**ABSTRACT**  
Document-level information extraction (IE) is a crucial task in natural language processing (NLP). This paper conducts a systematic review of recent document-level IE literature. In addition, we conduct a thorough error analysis with current state-of-the-art algorithms and identify their limitations as well as the remaining challenges for the task of document-level IE. According to our findings, labeling noises, entity coreference resolution, and lack of reasoning, severely affect the performance of document-level IE. The objective of this survey paper is to provide more insights and help NLP researchers to further enhance document-level IE performance.

{{</citation>}}


### (42/61) ChEDDAR: Student-ChatGPT Dialogue in EFL Writing Education (Jieun Han et al., 2023)

{{<citation>}}

Jieun Han, Haneul Yoo, Junho Myung, Minsun Kim, Tak Yeon Lee, So-Yeon Ahn, Alice Oh. (2023)  
**ChEDDAR: Student-ChatGPT Dialogue in EFL Writing Education**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2309.13243v1)  

---


**ABSTRACT**  
The integration of generative AI in education is expanding, yet empirical analyses of large-scale, real-world interactions between students and AI systems still remain limited. In this study, we present ChEDDAR, ChatGPT & EFL Learner's Dialogue Dataset As Revising an essay, which is collected from a semester-long longitudinal experiment involving 212 college students enrolled in English as Foreign Langauge (EFL) writing courses. The students were asked to revise their essays through dialogues with ChatGPT. ChEDDAR includes a conversation log, utterance-level essay edit history, self-rated satisfaction, and students' intent, in addition to session-level pre-and-post surveys documenting their objectives and overall experiences. We analyze students' usage patterns and perceptions regarding generative AI with respect to their intent and satisfaction. As a foundational step, we establish baseline results for two pivotal tasks in task-oriented dialogue systems within educational contexts: intent detection and satisfaction estimation. We finally suggest further research to refine the integration of generative AI into education settings, outlining potential scenarios utilizing ChEDDAR. ChEDDAR is publicly available at https://github.com/zeunie/ChEDDAR.

{{</citation>}}


### (43/61) User Simulation with Large Language Models for Evaluating Task-Oriented Dialogue (Sam Davidson et al., 2023)

{{<citation>}}

Sam Davidson, Salvatore Romeo, Raphael Shu, James Gung, Arshit Gupta, Saab Mansour, Yi Zhang. (2023)  
**User Simulation with Large Language Models for Evaluating Task-Oriented Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13233v1)  

---


**ABSTRACT**  
One of the major impediments to the development of new task-oriented dialogue (TOD) systems is the need for human evaluation at multiple stages and iterations of the development process. In an effort to move toward automated evaluation of TOD, we propose a novel user simulator built using recently developed large pretrained language models (LLMs). In order to increase the linguistic diversity of our system relative to the related previous work, we do not fine-tune the LLMs used by our system on existing TOD datasets; rather we use in-context learning to prompt the LLMs to generate robust and linguistically diverse output with the goal of simulating the behavior of human interlocutors. Unlike previous work, which sought to maximize goal success rate (GSR) as the primary metric of simulator performance, our goal is a system which achieves a GSR similar to that observed in human interactions with TOD systems. Using this approach, our current simulator is effectively able to interact with several TOD systems, especially on single-intent conversational goals, while generating lexically and syntactically diverse output relative to previous simulators that rely upon fine-tuned models. Finally, we collect a Human2Bot dataset of humans interacting with the same TOD systems with which we experimented in order to better quantify these achievements.

{{</citation>}}


### (44/61) NJUNLP's Participation for the WMT2023 Quality Estimation Shared Task (Xiang Geng et al., 2023)

{{<citation>}}

Xiang Geng, Zhejian Lai, Yu Zhang, Shimin Tao, Hao Yang, Jiajun Chen, Shujian Huang. (2023)  
**NJUNLP's Participation for the WMT2023 Quality Estimation Shared Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.13230v1)  

---


**ABSTRACT**  
We introduce the submissions of the NJUNLP team to the WMT 2023 Quality Estimation (QE) shared task. Our team submitted predictions for the English-German language pair on all two sub-tasks: (i) sentence- and word-level quality prediction; and (ii) fine-grained error span detection. This year, we further explore pseudo data methods for QE based on NJUQE framework (https://github.com/NJUNLP/njuqe). We generate pseudo MQM data using parallel data from the WMT translation task. We pre-train the XLMR large model on pseudo QE data, then fine-tune it on real QE data. At both stages, we jointly learn sentence-level scores and word-level tags. Empirically, we conduct experiments to find the key hyper-parameters that improve the performance. Technically, we propose a simple method that covert the word-level outputs to fine-grained error span results. Overall, our models achieved the best results in English-German for both word-level and fine-grained error span detection sub-tasks by a considerable margin.

{{</citation>}}


### (45/61) Hindi to English: Transformer-Based Neural Machine Translation (Kavit Gangar et al., 2023)

{{<citation>}}

Kavit Gangar, Hardik Ruparel, Shreyas Lele. (2023)  
**Hindi to English: Transformer-Based Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, Machine Translation, NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2309.13222v1)  

---


**ABSTRACT**  
Machine Translation (MT) is one of the most prominent tasks in Natural Language Processing (NLP) which involves the automatic conversion of texts from one natural language to another while preserving its meaning and fluency. Although the research in machine translation has been going on since multiple decades, the newer approach of integrating deep learning techniques in natural language processing has led to significant improvements in the translation quality. In this paper, we have developed a Neural Machine Translation (NMT) system by training the Transformer model to translate texts from Indian Language Hindi to English. Hindi being a low resource language has made it difficult for neural networks to understand the language thereby leading to a slow growth in the development of neural machine translators. Thus, to address this gap, we implemented back-translation to augment the training data and for creating the vocabulary, we experimented with both word and subword level tokenization using Byte Pair Encoding (BPE) thereby ending up training the Transformer in 10 different configurations. This led us to achieve a state-of-the-art BLEU score of 24.53 on the test set of IIT Bombay English-Hindi Corpus in one of the configurations.

{{</citation>}}


## cs.AR (1)



### (46/61) AxOMaP: Designing FPGA-based Approximate Arithmetic Operators using Mathematical Programming (Siva Satyendra Sahoo et al., 2023)

{{<citation>}}

Siva Satyendra Sahoo, Salim Ullah, Akash Kumar. (2023)  
**AxOMaP: Designing FPGA-based Approximate Arithmetic Operators using Mathematical Programming**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs.AR, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13445v1)  

---


**ABSTRACT**  
With the increasing application of machine learning (ML) algorithms in embedded systems, there is a rising necessity to design low-cost computer arithmetic for these resource-constrained systems. As a result, emerging models of computation, such as approximate and stochastic computing, that leverage the inherent error-resilience of such algorithms are being actively explored for implementing ML inference on resource-constrained systems. Approximate computing (AxC) aims to provide disproportionate gains in the power, performance, and area (PPA) of an application by allowing some level of reduction in its behavioral accuracy (BEHAV). Using approximate operators (AxOs) for computer arithmetic forms one of the more prevalent methods of implementing AxC. AxOs provide the additional scope for finer granularity of optimization, compared to only precision scaling of computer arithmetic. To this end, designing platform-specific and cost-efficient approximate operators forms an important research goal. Recently, multiple works have reported using AI/ML-based approaches for synthesizing novel FPGA-based AxOs. However, most of such works limit usage of AI/ML to designing ML-based surrogate functions used during iterative optimization processes. To this end, we propose a novel data analysis-driven mathematical programming-based approach to synthesizing approximate operators for FPGAs. Specifically, we formulate mixed integer quadratically constrained programs based on the results of correlation analysis of the characterization data and use the solutions to enable a more directed search approach for evolutionary optimization algorithms. Compared to traditional evolutionary algorithms-based optimization, we report up to 21% improvement in the hypervolume, for joint optimization of PPA and BEHAV, in the design of signed 8-bit multipliers.

{{</citation>}}


## cs.CR (1)



### (47/61) Moving Target Defense based Secured Network Slicing System in the O-RAN Architecture (Mojdeh Karbalaee Motalleb et al., 2023)

{{<citation>}}

Mojdeh Karbalaee Motalleb, Chafika Benzaïd, Tarik Taleb, Vahid Shah-Mansouri. (2023)  
**Moving Target Defense based Secured Network Slicing System in the O-RAN Architecture**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.13444v1)  

---


**ABSTRACT**  
The open radio access network (O-RAN) architecture's native virtualization and embedded intelligence facilitate RAN slicing and enable comprehensive end-to-end services in post-5G networks. However, any vulnerabilities could harm security. Therefore, artificial intelligence (AI) and machine learning (ML) security threats can even threaten O-RAN benefits. This paper proposes a novel approach to estimating the optimal number of predefined VNFs for each slice while addressing secure AI/ML methods for dynamic service admission control and power minimization in the O-RAN architecture. We solve this problem on two-time scales using mathematical methods for determining the predefined number of VNFs on a large time scale and the proximal policy optimization (PPO), a Deep Reinforcement Learning algorithm, for solving dynamic service admission control and power minimization for different slices on a small-time scale. To secure the ML system for O-RAN, we implement a moving target defense (MTD) strategy to prevent poisoning attacks by adding uncertainty to the system. Our experimental results show that the proposed PPO-based service admission control approach achieves an admission rate above 80\% and that the MTD strategy effectively strengthens the robustness of the PPO method against adversarial attacks.

{{</citation>}}


## eess.IV (2)



### (48/61) WS-YOLO: Weakly Supervised Yolo Network for Surgical Tool Localization in Endoscopic Videos (Rongfeng Wei et al., 2023)

{{<citation>}}

Rongfeng Wei, Jinlin Wu, You Pang, Zhen Chen. (2023)  
**WS-YOLO: Weakly Supervised Yolo Network for Surgical Tool Localization in Endoscopic Videos**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Yolo  
[Paper Link](http://arxiv.org/abs/2309.13404v2)  

---


**ABSTRACT**  
Being able to automatically detect and track surgical instruments in endoscopic video recordings would allow for many useful applications that could transform different aspects of surgery. In robot-assisted surgery, the potentially informative data like categories of surgical tool can be captured, which is sparse, full of noise and without spatial information. We proposed a Weakly Supervised Yolo Network (WS-YOLO) for Surgical Tool Localization in Endoscopic Videos, to generate fine-grained semantic information with location and category from coarse-grained semantic information outputted by the da Vinci surgical robot, which significantly diminished the necessary human annotation labor while striking an optimal balance between the quantity of manually annotated data and detection performance. The source code is available at https://github.com/Breezewrf/Weakly-Supervised-Yolov8.

{{</citation>}}


### (49/61) A mirror-Unet architecture for PET/CT lesion segmentation (Yamila Rotstein Habarnau et al., 2023)

{{<citation>}}

Yamila Rotstein Habarnau, Mauro Namías. (2023)  
**A mirror-Unet architecture for PET/CT lesion segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13398v1)  

---


**ABSTRACT**  
Automatic lesion detection and segmentation from [${}^{18}$F]FDG PET/CT scans is a challenging task, due to the diversity of shapes, sizes, FDG uptake and location they may present, besides the fact that physiological uptake is also present on healthy tissues. In this work, we propose a deep learning method aimed at the segmentation of oncologic lesions, based on a combination of two UNet-3D branches. First, one of the network's branches is trained to segment a group of tissues from CT images. The other branch is trained to segment the lesions from PET images, combining on the bottleneck the embedded information of CT branch, already trained. We trained and validated our networks on the AutoPET MICCAI 2023 Challenge dataset. Our code is available at: https://github.com/yrotstein/AutoPET2023_Mv1.

{{</citation>}}


## cs.AI (1)



### (50/61) D-Separation for Causal Self-Explanation (Wei Liu et al., 2023)

{{<citation>}}

Wei Liu, Jun Wang, Haozhao Wang, Ruixuan Li, Zhiying Deng, YuanKai Zhang, Yang Qiu. (2023)  
**D-Separation for Causal Self-Explanation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.13391v1)  

---


**ABSTRACT**  
Rationalization is a self-explaining framework for NLP models. Conventional work typically uses the maximum mutual information (MMI) criterion to find the rationale that is most indicative of the target label. However, this criterion can be influenced by spurious features that correlate with the causal rationale or the target label. Instead of attempting to rectify the issues of the MMI criterion, we propose a novel criterion to uncover the causal rationale, termed the Minimum Conditional Dependence (MCD) criterion, which is grounded on our finding that the non-causal features and the target label are \emph{d-separated} by the causal rationale. By minimizing the dependence between the unselected parts of the input and the target label conditioned on the selected rationale candidate, all the causes of the label are compelled to be selected. In this study, we employ a simple and practical measure of dependence, specifically the KL-divergence, to validate our proposed MCD criterion. Empirically, we demonstrate that MCD improves the F1 score by up to $13.7\%$ compared to previous state-of-the-art MMI-based methods. Our code is available at: \url{https://github.com/jugechengzi/Rationalization-MCD}.

{{</citation>}}


## cs.IR (2)



### (51/61) Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning (Zihua Si et al., 2023)

{{<citation>}}

Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, Jun Xu. (2023)  
**Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.13375v1)  

---


**ABSTRACT**  
The retrieval phase is a vital component in recommendation systems, requiring the model to be effective and efficient. Recently, generative retrieval has become an emerging paradigm for document retrieval, showing notable performance. These methods enjoy merits like being end-to-end differentiable, suggesting their viability in recommendation. However, these methods fall short in efficiency and effectiveness for large-scale recommendations. To obtain efficiency and effectiveness, this paper introduces a generative retrieval framework, namely SEATER, which learns SEmAntic Tree-structured item identifiERs via contrastive learning. Specifically, we employ an encoder-decoder model to extract user interests from historical behaviors and retrieve candidates via tree-structured item identifiers. SEATER devises a balanced k-ary tree structure of item identifiers, allocating semantic space to each token individually. This strategy maintains semantic consistency within the same level, while distinct levels correlate to varying semantic granularities. This structure also maintains consistent and fast inference speed for all items. Considering the tree structure, SEATER learns identifier tokens' semantics, hierarchical relationships, and inter-token dependencies. To achieve this, we incorporate two contrastive learning tasks with the generation task to optimize both the model and identifiers. The infoNCE loss aligns the token embeddings based on their hierarchical positions. The triplet loss ranks similar identifiers in desired orders. In this way, SEATER achieves both efficiency and effectiveness. Extensive experiments on three public datasets and an industrial dataset have demonstrated that SEATER outperforms state-of-the-art models significantly.

{{</citation>}}


### (52/61) Model-enhanced Vector Index (Hailin Zhang et al., 2023)

{{<citation>}}

Hailin Zhang, Yujing Wang, Qi Chen, Ruiheng Chang, Ting Zhang, Ziming Miao, Yingyan Hou, Yang Ding, Xupeng Miao, Haonan Wang, Bochen Pang, Yuefeng Zhan, Hao Sun, Weiwei Deng, Qi Zhang, Fan Yang, Xing Xie, Mao Yang, Bin Cui. (2023)  
**Model-enhanced Vector Index**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Embedding, Quantization  
[Paper Link](http://arxiv.org/abs/2309.13335v1)  

---


**ABSTRACT**  
Embedding-based retrieval methods construct vector indices to search for document representations that are most similar to the query representations. They are widely used in document retrieval due to low latency and decent recall performance. Recent research indicates that deep retrieval solutions offer better model quality, but are hindered by unacceptable serving latency and the inability to support document updates. In this paper, we aim to enhance the vector index with end-to-end deep generative models, leveraging the differentiable advantages of deep retrieval models while maintaining desirable serving efficiency. We propose Model-enhanced Vector Index (MEVI), a differentiable model-enhanced index empowered by a twin-tower representation model. MEVI leverages a Residual Quantization (RQ) codebook to bridge the sequence-to-sequence deep retrieval and embedding-based models. To substantially reduce the inference time, instead of decoding the unique document ids in long sequential steps, we first generate some semantic virtual cluster ids of candidate documents in a small number of steps, and then leverage the well-adapted embedding vectors to further perform a fine-grained search for the relevant documents in the candidate virtual clusters. We empirically show that our model achieves better performance on the commonly used academic benchmarks MSMARCO Passage and Natural Questions, with comparable serving latency to dense retrieval solutions.

{{</citation>}}


## cs.SD (1)



### (53/61) Asca: less audio data is more insightful (Xiang Li et al., 2023)

{{<citation>}}

Xiang Li, Junhao Chen, Chao Li, Hongwu Lv. (2023)  
**Asca: less audio data is more insightful**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.13373v1)  

---


**ABSTRACT**  
Audio recognition in specialized areas such as birdsong and submarine acoustics faces challenges in large-scale pre-training due to the limitations in available samples imposed by sampling environments and specificity requirements. While the Transformer model excels in audio recognition, its dependence on vast amounts of data becomes restrictive in resource-limited settings. Addressing this, we introduce the Audio Spectrogram Convolution Attention (ASCA) based on CoAtNet, integrating a Transformer-convolution hybrid architecture, novel network design, and attention techniques, further augmented with data enhancement and regularization strategies. On the BirdCLEF2023 and AudioSet(Balanced), ASCA achieved accuracies of 81.2% and 35.1%, respectively, significantly outperforming competing methods. The unique structure of our model enriches output, enabling generalization across various audio detection tasks. Our code can be found at https://github.com/LeeCiang/ASCA.

{{</citation>}}


## physics.geo-ph (1)



### (54/61) Accelerating Particle and Fluid Simulations with Differentiable Graph Networks for Solving Forward and Inverse Problems (Krishna Kumar et al., 2023)

{{<citation>}}

Krishna Kumar, Yongjin Choi. (2023)  
**Accelerating Particle and Fluid Simulations with Differentiable Graph Networks for Solving Forward and Inverse Problems**  

---
Primary Category: physics.geo-ph  
Categories: I-2; I-6-8, cs-LG, physics-comp-ph, physics-geo-ph, physics.geo-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13348v1)  

---


**ABSTRACT**  
We leverage physics-embedded differentiable graph network simulators (GNS) to accelerate particulate and fluid simulations to solve forward and inverse problems. GNS represents the domain as a graph with particles as nodes and learned interactions as edges. Compared to modeling global dynamics, GNS enables learning local interaction laws through edge messages, improving its generalization to new environments. GNS achieves over 165x speedup for granular flow prediction compared to parallel CPU numerical simulations. We propose a novel hybrid GNS/Material Point Method (MPM) to accelerate forward simulations by minimizing error on a pure surrogate model by interleaving MPM in GNS rollouts to satisfy conservation laws and minimize errors achieving 24x speedup compared to pure numerical simulations. The differentiable GNS enables solving inverse problems through automatic differentiation, identifying material parameters that result in target runout distances. We demonstrate the ability of GNS to solve inverse problems by iteratively updating the friction angle (a material property) by computing the gradient of a loss function based on the final and target runouts, thereby identifying the friction angle that best matches the observed runout. The physics-embedded and differentiable simulators open an exciting new paradigm for AI-accelerated design, control, and optimization.

{{</citation>}}


## cs.NI (1)



### (55/61) Joint Explainability and Sensitivity-Aware Federated Deep Learning for Transparent 6G RAN Slicing (Swastika Roy et al., 2023)

{{<citation>}}

Swastika Roy, Farhad Rezazadeh, Hatim Chergui, Christos Verikoukis. (2023)  
**Joint Explainability and Sensitivity-Aware Federated Deep Learning for Transparent 6G RAN Slicing**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13325v1)  

---


**ABSTRACT**  
In recent years, wireless networks are evolving complex, which upsurges the use of zero-touch artificial intelligence (AI)-driven network automation within the telecommunication industry. In particular, network slicing, the most promising technology beyond 5G, would embrace AI models to manage the complex communication network. Besides, it is also essential to build the trustworthiness of the AI black boxes in actual deployment when AI makes complex resource management and anomaly detection. Inspired by closed-loop automation and Explainable Artificial intelligence (XAI), we design an Explainable Federated deep learning (FDL) model to predict per-slice RAN dropped traffic probability while jointly considering the sensitivity and explainability-aware metrics as constraints in such non-IID setup. In precise, we quantitatively validate the faithfulness of the explanations via the so-called attribution-based \emph{log-odds metric} that is included as a constraint in the run-time FL optimization task. Simulation results confirm its superiority over an unconstrained integrated-gradient (IG) \emph{post-hoc} FDL baseline.

{{</citation>}}


## cs.SI (1)



### (56/61) Multilevel User Credibility Assessment in Social Networks (Mohammad Moradi et al., 2023)

{{<citation>}}

Mohammad Moradi, Mostafa Haghir Chehreghani. (2023)  
**Multilevel User Credibility Assessment in Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2309.13305v1)  

---


**ABSTRACT**  
Online social networks are one of the largest platforms for disseminating both real and fake news. Many users on these networks, intentionally or unintentionally, spread harmful content, fake news, and rumors in fields such as politics and business. As a result, numerous studies have been conducted in recent years to assess the credibility of users. A shortcoming of most of existing methods is that they assess users by placing them in one of two categories, real or fake. However, in real-world applications it is usually more desirable to consider several levels of user credibility. Another shortcoming is that existing approaches only use a portion of important features, which downgrades their performance. In this paper, due to the lack of an appropriate dataset for multilevel user credibility assessment, first we design a method to collect data suitable to assess credibility at multiple levels. Then, we develop the MultiCred model that places users at one of several levels of credibility, based on a rich and diverse set of features extracted from users' profile, tweets and comments. MultiCred exploits deep language models to analyze textual data and deep neural models to process non-textual features. Our extensive experiments reveal that MultiCred considerably outperforms existing approaches, in terms of several accuracy measures.

{{</citation>}}


## cs.RO (2)



### (57/61) Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning (Zhehui Huang et al., 2023)

{{<citation>}}

Zhehui Huang, Zhaojing Yang, Rahul Krupani, Baskın Şenbaşlar, Sumeet Batra, Gaurav S. Sukhatme. (2023)  
**Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.13285v1)  

---


**ABSTRACT**  
End-to-end deep reinforcement learning (DRL) for quadrotor control promises many benefits -- easy deployment, task generalization and real-time execution capability. Prior end-to-end DRL-based methods have showcased the ability to deploy learned controllers onto single quadrotors or quadrotor teams maneuvering in simple, obstacle-free environments. However, the addition of obstacles increases the number of possible interactions exponentially, thereby increasing the difficulty of training RL policies. In this work, we propose an end-to-end DRL approach to control quadrotor swarms in environments with obstacles. We provide our agents a curriculum and a replay buffer of the clipped collision episodes to improve performance in obstacle-rich environments. We implement an attention mechanism to attend to the neighbor robots and obstacle interactions - the first successful demonstration of this mechanism on policies for swarm behavior deployed on severely compute-constrained hardware. Our work is the first work that demonstrates the possibility of learning neighbor-avoiding and obstacle-avoiding control policies trained with end-to-end DRL that transfers zero-shot to real quadrotors. Our approach scales to 32 robots with 80% obstacle density in simulation and 8 robots with 20% obstacle density in physical deployment. Video demonstrations are available on the project website at: https://sites.google.com/view/obst-avoid-swarm-rl.

{{</citation>}}


### (58/61) Pick Planning Strategies for Large-Scale Package Manipulation (Shuai Li et al., 2023)

{{<citation>}}

Shuai Li, Azarakhsh Keipour, Kevin Jamieson, Nicolas Hudson, Sicong Szhao, Charles Swan, Kostas Bekris. (2023)  
**Pick Planning Strategies for Large-Scale Package Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2309.13224v1)  

---


**ABSTRACT**  
Automating warehouse operations can reduce logistics overhead costs, ultimately driving down the final price for consumers, increasing the speed of delivery, and enhancing the resiliency to market fluctuations.   This extended abstract showcases a large-scale package manipulation from unstructured piles in Amazon Robotics' Robot Induction (Robin) fleet, which is used for picking and singulating up to 6 million packages per day and so far has manipulated over 2 billion packages. It describes the various heuristic methods developed over time and their successor, which utilizes a pick success predictor trained on real production data.   To the best of the authors' knowledge, this work is the first large-scale deployment of learned pick quality estimation methods in a real production system.

{{</citation>}}


## cs.SE (1)



### (59/61) Natural Language Processing for Requirements Formalization: How to Derive New Approaches? (Viju Sudhi et al., 2023)

{{<citation>}}

Viju Sudhi, Libin Kutty, Robin Gröpler. (2023)  
**Natural Language Processing for Requirements Formalization: How to Derive New Approaches?**  

---
Primary Category: cs.SE  
Categories: 68T50, 68N30, I-2-7; D-2-1, cs-CL, cs-SE, cs.SE  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.13272v1)  

---


**ABSTRACT**  
It is a long-standing desire of industry and research to automate the software development and testing process as much as possible. In this process, requirements engineering (RE) plays a fundamental role for all other steps that build on it. Model-based design and testing methods have been developed to handle the growing complexity and variability of software systems. However, major effort is still required to create specification models from a large set of functional requirements provided in natural language. Numerous approaches based on natural language processing (NLP) have been proposed in the literature to generate requirements models using mainly syntactic properties. Recent advances in NLP show that semantic quantities can also be identified and used to provide better assistance in the requirements formalization process. In this work, we present and discuss principal ideas and state-of-the-art methodologies from the field of NLP in order to guide the readers on how to create a set of rules and methods for the semi-automated formalization of requirements according to their specific use case and needs. We discuss two different approaches in detail and highlight the iterative development of rule sets. The requirements models are represented in a human- and machine-readable format in the form of pseudocode. The presented methods are demonstrated on two industrial use cases from the automotive and railway domains. It shows that using current pre-trained NLP models requires less effort to create a set of rules and can be easily adapted to specific use cases and domains. In addition, findings and shortcomings of this research area are highlighted and an outlook on possible future developments is given.

{{</citation>}}


## cs.GT (1)



### (60/61) Chunking Tasks for Present-Biased Agents (Joe Halpern et al., 2023)

{{<citation>}}

Joe Halpern, Aditya Saraf. (2023)  
**Chunking Tasks for Present-Biased Agents**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.13244v1)  

---


**ABSTRACT**  
Everyone puts things off sometimes. How can we combat this tendency to procrastinate? A well-known technique used by instructors is to break up a large project into more manageable chunks. But how should this be done best? Here we study the process of chunking using the graph-theoretic model of present bias introduced by Kleinberg and Oren (2014). We first analyze how to optimally chunk single edges within a task graph, given a limited number of chunks. We show that for edges on the shortest path, the optimal chunking makes initial chunks easy and later chunks progressively harder. For edges not on the shortest path, optimal chunking is significantly more complex, but we provide an efficient algorithm that chunks the edge optimally. We then use our optimal edge-chunking algorithm to optimally chunk task graphs. We show that with a linear number of chunks on each edge, the biased agent's cost can be exponentially lowered, to within a constant factor of the true cheapest path. Finally, we extend our model to the case where a task designer must chunk a graph for multiple types of agents simultaneously. The problem grows significantly more complex with even two types of agents, but we provide optimal graph chunking algorithms for two types. Our work highlights the efficacy of chunking as a means to combat present bias.

{{</citation>}}


## cs.IT (1)



### (61/61) Causal Reasoning: Charting a Revolutionary Course for Next-Generation AI-Native Wireless Networks (Christo Kurisummoottil Thomas et al., 2023)

{{<citation>}}

Christo Kurisummoottil Thomas, Christina Chaccour, Walid Saad, Merouane Debbah, Choong Seon Hong. (2023)  
**Causal Reasoning: Charting a Revolutionary Course for Next-Generation AI-Native Wireless Networks**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs.IT, math-IT  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.13223v1)  

---


**ABSTRACT**  
Despite the basic premise that next-generation wireless networks (e.g., 6G) will be artificial intelligence (AI)-native, to date, most existing efforts remain either qualitative or incremental extensions to existing ``AI for wireless'' paradigms. Indeed, creating AI-native wireless networks faces significant technical challenges due to the limitations of data-driven, training-intensive AI. These limitations include the black-box nature of the AI models, their curve-fitting nature, which can limit their ability to reason and adapt, their reliance on large amounts of training data, and the energy inefficiency of large neural networks. In response to these limitations, this article presents a comprehensive, forward-looking vision that addresses these shortcomings by introducing a novel framework for building AI-native wireless networks; grounded in the emerging field of causal reasoning. Causal reasoning, founded on causal discovery, causal representation learning, and causal inference, can help build explainable, reasoning-aware, and sustainable wireless networks. Towards fulfilling this vision, we first highlight several wireless networking challenges that can be addressed by causal discovery and representation, including ultra-reliable beamforming for terahertz (THz) systems, near-accurate physical twin modeling for digital twins, training data augmentation, and semantic communication. We showcase how incorporating causal discovery can assist in achieving dynamic adaptability, resilience, and cognition in addressing these challenges. Furthermore, we outline potential frameworks that leverage causal inference to achieve the overarching objectives of future-generation networks, including intent management, dynamic adaptability, human-level cognition, reasoning, and the critical element of time sensitivity.

{{</citation>}}
