---
draft: false
title: "arXiv @ 2023.07.02"
date: 2023-07-02
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.02"
    identifier: arxiv_20230702
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (11)](#cslg-11)
- [cs.CV (10)](#cscv-10)
- [cs.AI (2)](#csai-2)
- [eess.SY (1)](#eesssy-1)
- [cs.RO (2)](#csro-2)
- [cs.SE (3)](#csse-3)
- [cs.IR (2)](#csir-2)
- [cs.CL (8)](#cscl-8)
- [cs.IT (1)](#csit-1)
- [eess.IV (2)](#eessiv-2)
- [q-fin.PR (1)](#q-finpr-1)
- [cs.CR (1)](#cscr-1)

## cs.LG (11)



### (1/44) CLIMAX: An exploration of Classifier-Based Contrastive Explanations (Praharsh Nanavati et al., 2023)

{{<citation>}}

Praharsh Nanavati, Ranjitha Prasad. (2023)  
**CLIMAX: An exploration of Classifier-Based Contrastive Explanations**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00680v1)  

---


**ABSTRACT**  
Explainable AI is an evolving area that deals with understanding the decision making of machine learning models so that these models are more transparent, accountable, and understandable for humans. In particular, post-hoc model-agnostic interpretable AI techniques explain the decisions of a black-box ML model for a single instance locally, without the knowledge of the intrinsic nature of the ML model. Despite their simplicity and capability in providing valuable insights, existing approaches fail to deliver consistent and reliable explanations. Moreover, in the context of black-box classifiers, existing approaches justify the predicted class, but these methods do not ensure that the explanation scores strongly differ as compared to those of another class. In this work we propose a novel post-hoc model agnostic XAI technique that provides contrastive explanations justifying the classification of a black box classifier along with a reasoning as to why another class was not predicted. Our method, which we refer to as CLIMAX which is short for Contrastive Label-aware Influence-based Model Agnostic XAI, is based on local classifiers . In order to ensure model fidelity of the explainer, we require the perturbations to be such that it leads to a class-balanced surrogate dataset. Towards this, we employ a label-aware surrogate data generation method based on random oversampling and Gaussian Mixture Model sampling. Further, we propose influence subsampling in order to retaining effective samples and hence ensure sample complexity. We show that we achieve better consistency as compared to baselines such as LIME, BayLIME, and SLIME. We also depict results on textual and image based datasets, where we generate contrastive explanations for any black-box classification model where one is able to only query the class probabilities for an instance of interest.

{{</citation>}}


### (2/44) Fraunhofer SIT at CheckThat! 2023: Mixing Single-Modal Classifiers to Estimate the Check-Worthiness of Multi-Modal Tweets (Raphael Frick et al., 2023)

{{<citation>}}

Raphael Frick, Inna Vogel. (2023)  
**Fraunhofer SIT at CheckThat! 2023: Mixing Single-Modal Classifiers to Estimate the Check-Worthiness of Multi-Modal Tweets**  

---
Primary Category: cs.LG
Categories: cs-CL, cs-LG, cs-SI, cs.LG  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2307.00610v1)  

---


**ABSTRACT**  
The option of sharing images, videos and audio files on social media opens up new possibilities for distinguishing between false information and fake news on the Internet. Due to the vast amount of data shared every second on social media, not all data can be verified by a computer or a human expert. Here, a check-worthiness analysis can be used as a first step in the fact-checking pipeline and as a filtering mechanism to improve efficiency. This paper proposes a novel way of detecting the check-worthiness in multi-modal tweets. It takes advantage of two classifiers, each trained on a single modality. For image data, extracting the embedded text with an OCR analysis has shown to perform best. By combining the two classifiers, the proposed solution was able to place first in the CheckThat! 2023 Task 1A with an F1 score of 0.7297 achieved on the private test set.

{{</citation>}}


### (3/44) Conditionally Invariant Representation Learning for Disentangling Cellular Heterogeneity (Hananeh Aliee et al., 2023)

{{<citation>}}

Hananeh Aliee, Ferdinand Kapl, Soroor Hediyeh-Zadeh, Fabian J. Theis. (2023)  
**Conditionally Invariant Representation Learning for Disentangling Cellular Heterogeneity**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.00558v1)  

---


**ABSTRACT**  
This paper presents a novel approach that leverages domain variability to learn representations that are conditionally invariant to unwanted variability or distractors. Our approach identifies both spurious and invariant latent features necessary for achieving accurate reconstruction by placing distinct conditional priors on latent features. The invariant signals are disentangled from noise by enforcing independence which facilitates the construction of an interpretable model with a causal semantic. By exploiting the interplay between data domains and labels, our method simultaneously identifies invariant features and builds invariant predictors. We apply our method to grand biological challenges, such as data integration in single-cell genomics with the aim of capturing biological variations across datasets with many samples, obtained from different conditions or multiple laboratories. Our approach allows for the incorporation of specific biological mechanisms, including gene programs, disease states, or treatment conditions into the data integration process, bridging the gap between the theoretical assumptions and real biological applications. Specifically, the proposed approach helps to disentangle biological signals from data biases that are unrelated to the target task or the causal explanation of interest. Through extensive benchmarking using large-scale human hematopoiesis and human lung cancer data, we validate the superiority of our approach over existing methods and demonstrate that it can empower deeper insights into cellular heterogeneity and the identification of disease cell states.

{{</citation>}}


### (4/44) Adaptive reinforcement learning of multi-agent ethically-aligned behaviours: the QSOM and QDSOM algorithms (Rémy Chaput et al., 2023)

{{<citation>}}

Rémy Chaput, Olivier Boissier, Mathieu Guillermin. (2023)  
**Adaptive reinforcement learning of multi-agent ethically-aligned behaviours: the QSOM and QDSOM algorithms**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CY, cs-LG, cs-MA, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00552v1)  

---


**ABSTRACT**  
The numerous deployed Artificial Intelligence systems need to be aligned with our ethical considerations. However, such ethical considerations might change as time passes: our society is not fixed, and our social mores evolve. This makes it difficult for these AI systems; in the Machine Ethics field especially, it has remained an under-studied challenge. In this paper, we present two algorithms, named QSOM and QDSOM, which are able to adapt to changes in the environment, and especially in the reward function, which represents the ethical considerations that we want these systems to be aligned with. They associate the well-known Q-Table to (Dynamic) Self-Organizing Maps to handle the continuous and multi-dimensional state and action spaces. We evaluate them on a use-case of multi-agent energy repartition within a small Smart Grid neighborhood, and prove their ability to adapt, and their higher performance compared to baseline Reinforcement Learning algorithms.

{{</citation>}}


### (5/44) Is Risk-Sensitive Reinforcement Learning Properly Resolved? (Ruiwen Zhou et al., 2023)

{{<citation>}}

Ruiwen Zhou, Minghuan Liu, Kan Ren, Xufang Luo, Weinan Zhang, Dongsheng Li. (2023)  
**Is Risk-Sensitive Reinforcement Learning Properly Resolved?**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00547v1)  

---


**ABSTRACT**  
Due to the nature of risk management in learning applicable policies, risk-sensitive reinforcement learning (RSRL) has been realized as an important direction. RSRL is usually achieved by learning risk-sensitive objectives characterized by various risk measures, under the framework of distributional reinforcement learning. However, it remains unclear if the distributional Bellman operator properly optimizes the RSRL objective in the sense of risk measures. In this paper, we prove that the existing RSRL methods do not achieve unbiased optimization and can not guarantee optimality or even improvements regarding risk measures over accumulated return distributions. To remedy this issue, we further propose a novel algorithm, namely Trajectory Q-Learning (TQL), for RSRL problems with provable convergence to the optimal policy. Based on our new learning architecture, we are free to introduce a general and practical implementation for different risk measures to learn disparate risk-sensitive policies. In the experiments, we verify the learnability of our algorithm and show how our method effectively achieves better performances toward risk-sensitive objectives.

{{</citation>}}


### (6/44) Collaborative Policy Learning for Dynamic Scheduling Tasks in Cloud-Edge-Terminal IoT Networks Using Federated Reinforcement Learning (Do-Yup Kim et al., 2023)

{{<citation>}}

Do-Yup Kim, Da-Eun Lee, Ji-Wan Kim, Hyun-Suk Lee. (2023)  
**Collaborative Policy Learning for Dynamic Scheduling Tasks in Cloud-Edge-Terminal IoT Networks Using Federated Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: 68M20, 68T05, 68T07, C-2-1; C-2-4; I-2-8; I-2-11, cs-AI, cs-DC, cs-LG, cs.LG, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00541v1)  

---


**ABSTRACT**  
In this paper, we examine cloud-edge-terminal IoT networks, where edges undertake a range of typical dynamic scheduling tasks. In these IoT networks, a central policy for each task can be constructed at a cloud server. The central policy can be then used by the edges conducting the task, thereby mitigating the need for them to learn their own policy from scratch. Furthermore, this central policy can be collaboratively learned at the cloud server by aggregating local experiences from the edges, thanks to the hierarchical architecture of the IoT networks. To this end, we propose a novel collaborative policy learning framework for dynamic scheduling tasks using federated reinforcement learning. For effective learning, our framework adaptively selects the tasks for collaborative learning in each round, taking into account the need for fairness among tasks. In addition, as a key enabler of the framework, we propose an edge-agnostic policy structure that enables the aggregation of local policies from different edges. We then provide the convergence analysis of the framework. Through simulations, we demonstrate that our proposed framework significantly outperforms the approaches without collaborative policy learning. Notably, it accelerates the learning speed of the policies and allows newly arrived edges to adapt to their tasks more easily.

{{</citation>}}


### (7/44) Shared Growth of Graph Neural Networks via Free-direction Knowledge Distillation (Kaituo Feng et al., 2023)

{{<citation>}}

Kaituo Feng, Yikun Miao, Changsheng Li, Ye Yuan, Guoren Wang. (2023)  
**Shared Growth of Graph Neural Networks via Free-direction Knowledge Distillation**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.00534v2)  

---


**ABSTRACT**  
Knowledge distillation (KD) has shown to be effective to boost the performance of graph neural networks (GNNs), where the typical objective is to distill knowledge from a deeper teacher GNN into a shallower student GNN. However, it is often quite challenging to train a satisfactory deeper GNN due to the well-known over-parametrized and over-smoothing issues, leading to invalid knowledge transfer in practical applications. In this paper, we propose the first Free-direction Knowledge Distillation framework via reinforcement learning for GNNs, called FreeKD, which is no longer required to provide a deeper well-optimized teacher GNN. Our core idea is to collaboratively learn two shallower GNNs in an effort to exchange knowledge between them via reinforcement learning in a hierarchical way. As we observe that one typical GNN model often exhibits better and worse performances at different nodes during training, we devise a dynamic and free-direction knowledge transfer strategy that involves two levels of actions: 1) node-level action determines the directions of knowledge transfer between the corresponding nodes of two networks; and then 2) structure-level action determines which of the local structures generated by the node-level actions to be propagated. Furthermore, considering the diverse knowledge present in different GNNs when dealing with multi-view inputs, we introduce FreeKD++ as a solution to enable free-direction knowledge transfer among multiple shallow GNNs operating on multi-view inputs. Extensive experiments on five benchmark datasets demonstrate our approaches outperform the base GNNs in a large margin, and shows their efficacy to various GNNs. More surprisingly, our FreeKD has comparable or even better performance than traditional KD algorithms that distill knowledge from a deeper and stronger teacher GNN.

{{</citation>}}


### (8/44) Classifying World War II Era Ciphers with Machine Learning (Brooke Dalton et al., 2023)

{{<citation>}}

Brooke Dalton, Mark Stamp. (2023)  
**Classifying World War II Era Ciphers with Machine Learning**  

---
Primary Category: cs.LG
Categories: cs-CR, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.00501v1)  

---


**ABSTRACT**  
We determine the accuracy with which machine learning and deep learning techniques can classify selected World War II era ciphers when only ciphertext is available. The specific ciphers considered are Enigma, M-209, Sigaba, Purple, and Typex. We experiment with three classic machine learning models, namely, Support Vector Machines (SVM), $k$-Nearest Neighbors ($k$-NN), and Random Forest (RF). We also experiment with four deep learning neural network-based models: Multi-Layer Perceptrons (MLP), Long Short-Term Memory (LSTM), Extreme Learning Machines (ELM), and Convolutional Neural Networks (CNN). Each model is trained on features consisting of histograms, digrams, and raw ciphertext letter sequences. Furthermore, the classification problem is considered under four distinct scenarios: Fixed plaintext with fixed keys, random plaintext with fixed keys, fixed plaintext with random keys, and random plaintext with random keys. Under the most realistic scenario, given 1000 characters per ciphertext, we are able to distinguish the ciphers with greater than 97% accuracy. In addition, we consider the accuracy of a subset of the learning techniques as a function of the length of the ciphertext messages. Somewhat surprisingly, our classic machine learning models perform at least as well as our deep learning models. We also find that ciphers that are more similar in design are somewhat more challenging to distinguish, but not as difficult as might be expected.

{{</citation>}}


### (9/44) Data-Free Quantization via Mixed-Precision Compensation without Fine-Tuning (Jun Chen et al., 2023)

{{<citation>}}

Jun Chen, Shipeng Bai, Tianxin Huang, Mengmeng Wang, Guanzhong Tian, Yong Liu. (2023)  
**Data-Free Quantization via Mixed-Precision Compensation without Fine-Tuning**  

---
Primary Category: cs.LG
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2307.00498v1)  

---


**ABSTRACT**  
Neural network quantization is a very promising solution in the field of model compression, but its resulting accuracy highly depends on a training/fine-tuning process and requires the original data. This not only brings heavy computation and time costs but also is not conducive to privacy and sensitive information protection. Therefore, a few recent works are starting to focus on data-free quantization. However, data-free quantization does not perform well while dealing with ultra-low precision quantization. Although researchers utilize generative methods of synthetic data to address this problem partially, data synthesis needs to take a lot of computation and time. In this paper, we propose a data-free mixed-precision compensation (DF-MPC) method to recover the performance of an ultra-low precision quantized model without any data and fine-tuning process. By assuming the quantized error caused by a low-precision quantized layer can be restored via the reconstruction of a high-precision quantized layer, we mathematically formulate the reconstruction loss between the pre-trained full-precision model and its layer-wise mixed-precision quantized model. Based on our formulation, we theoretically deduce the closed-form solution by minimizing the reconstruction loss of the feature maps. Since DF-MPC does not require any original/synthetic data, it is a more efficient method to approximate the full-precision model. Experimentally, our DF-MPC is able to achieve higher accuracy for an ultra-low precision quantized model compared to the recent methods without any data and fine-tuning process.

{{</citation>}}


### (10/44) STG4Traffic: A Survey and Benchmark of Spatial-Temporal Graph Neural Networks for Traffic Prediction (Xunlian Luo et al., 2023)

{{<citation>}}

Xunlian Luo, Chunjiang Zhu, Detian Zhang, Qing Li. (2023)  
**STG4Traffic: A Survey and Benchmark of Spatial-Temporal Graph Neural Networks for Traffic Prediction**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.00495v1)  

---


**ABSTRACT**  
Traffic prediction has been an active research topic in the domain of spatial-temporal data mining. Accurate real-time traffic prediction is essential to improve the safety, stability, and versatility of smart city systems, i.e., traffic control and optimal routing. The complex and highly dynamic spatial-temporal dependencies make effective predictions still face many challenges. Recent studies have shown that spatial-temporal graph neural networks exhibit great potential applied to traffic prediction, which combines sequential models with graph convolutional networks to jointly model temporal and spatial correlations. However, a survey study of graph learning, spatial-temporal graph models for traffic, as well as a fair comparison of baseline models are pending and unavoidable issues. In this paper, we first provide a systematic review of graph learning strategies and commonly used graph convolution algorithms. Then we conduct a comprehensive analysis of the strengths and weaknesses of recently proposed spatial-temporal graph network models. Furthermore, we build a study called STG4Traffic using the deep learning framework PyTorch to establish a standardized and scalable benchmark on two types of traffic datasets. We can evaluate their performance by personalizing the model settings with uniform metrics. Finally, we point out some problems in the current study and discuss future directions. Source codes are available at https://github.com/trainingl/STG4Traffic.

{{</citation>}}


### (11/44) Fourier-Mixed Window Attention: Accelerating Informer for Long Sequence Time-Series Forecasting (Nhat Thanh Tran et al., 2023)

{{<citation>}}

Nhat Thanh Tran, Jack Xin. (2023)  
**Fourier-Mixed Window Attention: Accelerating Informer for Long Sequence Time-Series Forecasting**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.00493v1)  

---


**ABSTRACT**  
We study a fast local-global window-based attention method to accelerate Informer for long sequence time-series forecasting. While window attention is local and a considerable computational saving, it lacks the ability to capture global token information which is compensated by a subsequent Fourier transform block. Our method, named FWin, does not rely on query sparsity hypothesis and an empirical approximation underlying the ProbSparse attention of Informer. Through experiments on univariate and multivariate datasets, we show that FWin transformers improve the overall prediction accuracies of Informer while accelerating its inference speeds by 40 to 50 %. We also show in a nonlinear regression model that a learned FWin type attention approaches or even outperforms softmax full attention based on key vectors extracted from an Informer model's full attention layer acting on time series data.

{{</citation>}}


## cs.CV (10)



### (12/44) Pay Attention to the Atlas: Atlas-Guided Test-Time Adaptation Method for Robust 3D Medical Image Segmentation (Jingjie Guo et al., 2023)

{{<citation>}}

Jingjie Guo, Weitong Zhang, Matthew Sinclair, Daniel Rueckert, Chen Chen. (2023)  
**Pay Attention to the Atlas: Atlas-Guided Test-Time Adaptation Method for Robust 3D Medical Image Segmentation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.00676v1)  

---


**ABSTRACT**  
Convolutional neural networks (CNNs) often suffer from poor performance when tested on target data that differs from the training (source) data distribution, particularly in medical imaging applications where variations in imaging protocols across different clinical sites and scanners lead to different imaging appearances. However, re-accessing source training data for unsupervised domain adaptation or labeling additional test data for model fine-tuning can be difficult due to privacy issues and high labeling costs, respectively. To solve this problem, we propose a novel atlas-guided test-time adaptation (TTA) method for robust 3D medical image segmentation, called AdaAtlas. AdaAtlas only takes one single unlabeled test sample as input and adapts the segmentation network by minimizing an atlas-based loss. Specifically, the network is adapted so that its prediction after registration is aligned with the learned atlas in the atlas space, which helps to reduce anatomical segmentation errors at test time. In addition, different from most existing TTA methods which restrict the adaptation to batch normalization blocks in the segmentation network only, we further exploit the use of channel and spatial attention blocks for improved adaptability at test time. Extensive experiments on multiple datasets from different sites show that AdaAtlas with attention blocks adapted (AdaAtlas-Attention) achieves superior performance improvements, greatly outperforming other competitive TTA methods.

{{</citation>}}


### (13/44) CNN-BiLSTM model for English Handwriting Recognition: Comprehensive Evaluation on the IAM Dataset (Firat Kizilirmak et al., 2023)

{{<citation>}}

Firat Kizilirmak, Berrin Yanikoglu. (2023)  
**CNN-BiLSTM model for English Handwriting Recognition: Comprehensive Evaluation on the IAM Dataset**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.00664v1)  

---


**ABSTRACT**  
We present a CNN-BiLSTM system for the problem of offline English handwriting recognition, with extensive evaluations on the public IAM dataset, including the effects of model size, data augmentation and the lexicon. Our best model achieves 3.59\% CER and 9.44\% WER using CNN-BiLSTM network with CTC layer. Test time augmentation with rotation and shear transformations applied to the input image, is proposed to increase recognition of difficult cases and found to reduce the word error rate by 2.5\% points. We also conduct an error analysis of our proposed method on IAM dataset, show hard cases of handwriting images and explore samples with erroneous labels. We provide our source code as public-domain, to foster further research to encourage scientific reproducibility.

{{</citation>}}


### (14/44) More Synergy, Less Redundancy: Exploiting Joint Mutual Information for Self-Supervised Learning (Salman Mohamadi et al., 2023)

{{<citation>}}

Salman Mohamadi, Gianfranco Doretto, Donald A. Adjeroh. (2023)  
**More Synergy, Less Redundancy: Exploiting Joint Mutual Information for Self-Supervised Learning**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.00651v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) is now a serious competitor for supervised learning, even though it does not require data annotation. Several baselines have attempted to make SSL models exploit information about data distribution, and less dependent on the augmentation effect. However, there is no clear consensus on whether maximizing or minimizing the mutual information between representations of augmentation views practically contribute to improvement or degradation in performance of SSL models. This paper is a fundamental work where, we investigate role of mutual information in SSL, and reformulate the problem of SSL in the context of a new perspective on mutual information. To this end, we consider joint mutual information from the perspective of partial information decomposition (PID) as a key step in \textbf{reliable multivariate information measurement}. PID enables us to decompose joint mutual information into three important components, namely, unique information, redundant information and synergistic information. Our framework aims for minimizing the redundant information between views and the desired target representation while maximizing the synergistic information at the same time. Our experiments lead to a re-calibration of two redundancy reduction baselines, and a proposal for a new SSL training protocol. Extensive experimental results on multiple datasets and two downstream tasks show the effectiveness of this framework.

{{</citation>}}


### (15/44) Intra- & Extra-Source Exemplar-Based Style Synthesis for Improved Domain Generalization (Yumeng Li et al., 2023)

{{<citation>}}

Yumeng Li, Dan Zhang, Margret Keuper, Anna Khoreva. (2023)  
**Intra- & Extra-Source Exemplar-Based Style Synthesis for Improved Domain Generalization**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.00648v1)  

---


**ABSTRACT**  
The generalization with respect to domain shifts, as they frequently appear in applications such as autonomous driving, is one of the remaining big challenges for deep learning models. Therefore, we propose an exemplar-based style synthesis pipeline to improve domain generalization in semantic segmentation. Our method is based on a novel masked noise encoder for StyleGAN2 inversion. The model learns to faithfully reconstruct the image, preserving its semantic layout through noise prediction. Using the proposed masked noise encoder to randomize style and content combinations in the training set, i.e., intra-source style augmentation (ISSA) effectively increases the diversity of training data and reduces spurious correlation. As a result, we achieve up to $12.4\%$ mIoU improvements on driving-scene semantic segmentation under different types of data shifts, i.e., changing geographic locations, adverse weather conditions, and day to night. ISSA is model-agnostic and straightforwardly applicable with CNNs and Transformers. It is also complementary to other domain generalization techniques, e.g., it improves the recent state-of-the-art solution RobustNet by $3\%$ mIoU in Cityscapes to Dark Z\"urich. In addition, we demonstrate the strong plug-n-play ability of the proposed style synthesis pipeline, which is readily usable for extra-source exemplars e.g., web-crawled images, without any retraining or fine-tuning. Moreover, we study a new use case to indicate neural network's generalization capability by building a stylized proxy validation set. This application has significant practical sense for selecting models to be deployed in the open-world environment. Our code is available at \url{https://github.com/boschresearch/ISSA}.

{{</citation>}}


### (16/44) X-MLP: A Patch Embedding-Free MLP Architecture for Vision (Xinyue Wang et al., 2023)

{{<citation>}}

Xinyue Wang, Zhicheng Cai, Chenglei Peng. (2023)  
**X-MLP: A Patch Embedding-Free MLP Architecture for Vision**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.00592v1)  

---


**ABSTRACT**  
Convolutional neural networks (CNNs) and vision transformers (ViT) have obtained great achievements in computer vision. Recently, the research of multi-layer perceptron (MLP) architectures for vision have been popular again. Vision MLPs are designed to be independent from convolutions and self-attention operations. However, existing vision MLP architectures always depend on convolution for patch embedding. Thus we propose X-MLP, an architecture constructed absolutely upon fully connected layers and free from patch embedding. It decouples the features extremely and utilizes MLPs to interact the information across the dimension of width, height and channel independently and alternately. X-MLP is tested on ten benchmark datasets, all obtaining better performance than other vision MLP models. It even surpasses CNNs by a clear margin on various dataset. Furthermore, through mathematically restoring the spatial weights, we visualize the information communication between any couples of pixels in the feature map and observe the phenomenon of capturing long-range dependency.

{{</citation>}}


### (17/44) ClipSitu: Effectively Leveraging CLIP for Conditional Predictions in Situation Recognition (Debaditya Roy et al., 2023)

{{<citation>}}

Debaditya Roy, Dhruv Verma, Basura Fernando. (2023)  
**ClipSitu: Effectively Leveraging CLIP for Conditional Predictions in Situation Recognition**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.00586v1)  

---


**ABSTRACT**  
Situation Recognition is the task of generating a structured summary of what is happening in an image using an activity verb and the semantic roles played by actors and objects. In this task, the same activity verb can describe a diverse set of situations as well as the same actor or object category can play a diverse set of semantic roles depending on the situation depicted in the image. Hence model needs to understand the context of the image and the visual-linguistic meaning of semantic roles. Therefore, we leverage the CLIP foundational model that has learned the context of images via language descriptions. We show that deeper-and-wider multi-layer perceptron (MLP) blocks obtain noteworthy results for the situation recognition task by using CLIP image and text embedding features and it even outperforms the state-of-the-art CoFormer, a Transformer-based model, thanks to the external implicit visual-linguistic knowledge encapsulated by CLIP and the expressive power of modern MLP block designs. Motivated by this, we design a cross-attention-based Transformer using CLIP visual tokens that model the relation between textual roles and visual entities. Our cross-attention-based Transformer known as ClipSitu XTF outperforms existing state-of-the-art by a large margin of 14.1% on semantic role labelling (value) for top-1 accuracy using imSitu dataset. We will make the code publicly available.

{{</citation>}}


### (18/44) A MIL Approach for Anomaly Detection in Surveillance Videos from Multiple Camera Views (Silas Santiago Lopes Pereira et al., 2023)

{{<citation>}}

Silas Santiago Lopes Pereira, José Everardo Bessa Maia. (2023)  
**A MIL Approach for Anomaly Detection in Surveillance Videos from Multiple Camera Views**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.00562v1)  

---


**ABSTRACT**  
Occlusion and clutter are two scene states that make it difficult to detect anomalies in surveillance video. Furthermore, anomaly events are rare and, as a consequence, class imbalance and lack of labeled anomaly data are also key features of this task. Therefore, weakly supervised methods are heavily researched for this application. In this paper, we tackle these typical problems of anomaly detection in surveillance video by combining Multiple Instance Learning (MIL) to deal with the lack of labels and Multiple Camera Views (MC) to reduce occlusion and clutter effects. In the resulting MC-MIL algorithm we apply a multiple camera combined loss function to train a regression network with Sultani's MIL ranking function. To evaluate the MC-MIL algorithm first proposed here, the multiple camera PETS-2009 benchmark dataset was re-labeled for the anomaly detection task from multiple camera views. The result shows a significant performance improvement in F1 score compared to the single-camera configuration.

{{</citation>}}


### (19/44) Referring Video Object Segmentation with Inter-Frame Interaction and Cross-Modal Correlation (Meng Lan et al., 2023)

{{<citation>}}

Meng Lan, Fu Rong, Lefei Zhang. (2023)  
**Referring Video Object Segmentation with Inter-Frame Interaction and Cross-Modal Correlation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.00536v1)  

---


**ABSTRACT**  
Referring video object segmentation (RVOS) aims to segment the target object in a video sequence described by a language expression. Typical query-based methods process the video sequence in a frame-independent manner to reduce the high computational cost, which however affects the performance due to the lack of inter-frame interaction for temporal coherence modeling and spatio-temporal representation learning of the referred object. Besides, they directly adopt the raw and high-level sentence feature as the language queries to decode the visual features, where the weak correlation between visual and linguistic features also increases the difficulty of decoding the target information and limits the performance of the model. In this paper, we proposes a novel RVOS framework, dubbed IFIRVOS, to address these issues. Specifically, we design a plug-and-play inter-frame interaction module in the Transformer decoder to efficiently learn the spatio-temporal features of the referred object, so as to decode the object information in the video sequence more precisely and generate more accurate segmentation results. Moreover, we devise the vision-language interaction module before the multimodal Transformer to enhance the correlation between the visual and linguistic features, thus facilitating the process of decoding object information from visual features by language queries in Transformer decoder and improving the segmentation performance. Extensive experimental results on three benchmarks validate the superiority of our IFIRVOS over state-of-the-art methods and the effectiveness of our proposed modules.

{{</citation>}}


### (20/44) TopicFM+: Boosting Accuracy and Efficiency of Topic-Assisted Feature Matching (Khang Truong Giang et al., 2023)

{{<citation>}}

Khang Truong Giang, Soohwan Song, Sungho Jo. (2023)  
**TopicFM+: Boosting Accuracy and Efficiency of Topic-Assisted Feature Matching**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.00485v1)  

---


**ABSTRACT**  
This study tackles the challenge of image matching in difficult scenarios, such as scenes with significant variations or limited texture, with a strong emphasis on computational efficiency. Previous studies have attempted to address this challenge by encoding global scene contexts using Transformers. However, these approaches suffer from high computational costs and may not capture sufficient high-level contextual information, such as structural shapes or semantic instances. Consequently, the encoded features may lack discriminative power in challenging scenes. To overcome these limitations, we propose a novel image-matching method that leverages a topic-modeling strategy to capture high-level contexts in images. Our method represents each image as a multinomial distribution over topics, where each topic represents a latent semantic instance. By incorporating these topics, we can effectively capture comprehensive context information and obtain discriminative and high-quality features. Additionally, our method effectively matches features within corresponding semantic regions by estimating the covisible topics. To enhance the efficiency of feature matching, we have designed a network with a pooling-and-merging attention module. This module reduces computation by employing attention only on fixed-sized topics and small-sized features. Through extensive experiments, we have demonstrated the superiority of our method in challenging scenarios. Specifically, our method significantly reduces computational costs while maintaining higher image-matching accuracy compared to state-of-the-art methods. The code will be updated soon at https://github.com/TruongKhang/TopicFM

{{</citation>}}


### (21/44) Human-to-Human Interaction Detection (Zhenhua Wang et al., 2023)

{{<citation>}}

Zhenhua Wang, Kaining Ying, Jiajun Meng, Jifeng Ning, Cong Bai. (2023)  
**Human-to-Human Interaction Detection**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.00464v1)  

---


**ABSTRACT**  
A comprehensive understanding of interested human-to-human interactions in video streams, such as queuing, handshaking, fighting and chasing, is of immense importance to the surveillance of public security in regions like campuses, squares and parks. Different from conventional human interaction recognition, which uses choreographed videos as inputs, neglects concurrent interactive groups, and performs detection and recognition in separate stages, we introduce a new task named human-to-human interaction detection (HID). HID devotes to detecting subjects, recognizing person-wise actions, and grouping people according to their interactive relations, in one model. First, based on the popular AVA dataset created for action detection, we establish a new HID benchmark, termed AVA-Interaction (AVA-I), by adding annotations on interactive relations in a frame-by-frame manner. AVA-I consists of 85,254 frames and 86,338 interactive groups, and each image includes up to 4 concurrent interactive groups. Second, we present a novel baseline approach SaMFormer for HID, containing a visual feature extractor, a split stage which leverages a Transformer-based model to decode action instances and interactive groups, and a merging stage which reconstructs the relationship between instances and groups. All SaMFormer components are jointly trained in an end-to-end manner. Extensive experiments on AVA-I validate the superiority of SaMFormer over representative methods. The dataset and code will be made public to encourage more follow-up studies.

{{</citation>}}


## cs.AI (2)



### (22/44) Minimum Levels of Interpretability for Artificial Moral Agents (Avish Vijayaraghavan et al., 2023)

{{<citation>}}

Avish Vijayaraghavan, Cosmin Badea. (2023)  
**Minimum Levels of Interpretability for Artificial Moral Agents**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00660v1)  

---


**ABSTRACT**  
As artificial intelligence (AI) models continue to scale up, they are becoming more capable and integrated into various forms of decision-making systems. For models involved in moral decision-making, also known as artificial moral agents (AMA), interpretability provides a way to trust and understand the agent's internal reasoning mechanisms for effective use and error correction. In this paper, we provide an overview of this rapidly-evolving sub-field of AI interpretability, introduce the concept of the Minimum Level of Interpretability (MLI) and recommend an MLI for various types of agents, to aid their safe deployment in real-world settings.

{{</citation>}}


### (23/44) Neuro-Symbolic Sudoku Solver (Ashutosh Hathidara et al., 2023)

{{<citation>}}

Ashutosh Hathidara, Lalit Pandey. (2023)  
**Neuro-Symbolic Sudoku Solver**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-GT, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00653v1)  

---


**ABSTRACT**  
Deep Neural Networks have achieved great success in some of the complex tasks that humans can do with ease. These include image recognition/classification, natural language processing, game playing etc. However, modern Neural Networks fail or perform poorly when trained on tasks that can be solved easily using backtracking and traditional algorithms. Therefore, we use the architecture of the Neuro Logic Machine (NLM) and extend its functionality to solve a 9X9 game of Sudoku. To expand the application of NLMs, we generate a random grid of cells from a dataset of solved games and assign up to 10 new empty cells. The goal of the game is then to find a target value ranging from 1 to 9 and fill in the remaining empty cells while maintaining a valid configuration. In our study, we showcase an NLM which is capable of obtaining 100% accuracy for solving a Sudoku with empty cells ranging from 3 to 10. The purpose of this study is to demonstrate that NLMs can also be used for solving complex problems and games like Sudoku. We also analyze the behaviour of NLMs with a backtracking algorithm by comparing the convergence time using a graph plot on the same problem. With this study we show that Neural Logic Machines can be trained on the tasks that traditional Deep Learning architectures fail using Reinforcement Learning. We also aim to propose the importance of symbolic learning in explaining the systematicity in the hybrid model of NLMs.

{{</citation>}}


## eess.SY (1)



### (24/44) On Embedding B-Splines in Recursive State Estimation (Kailai Li, 2023)

{{<citation>}}

Kailai Li. (2023)  
**On Embedding B-Splines in Recursive State Estimation**  

---
Primary Category: eess.SY
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.00637v1)  

---


**ABSTRACT**  
We present a principled study on establishing a novel probabilistic framework for state estimation. B-splines are embedded in the state-space modeling as a continuous-time intermediate between the states of recurrent control points and asynchronous sensor measurements. Based thereon, the spline-embedded recursive estimation scheme is established w.r.t. common sensor fusion tasks, and the corresponding technique for modeling uncertain motion estimates is introduced. We evaluate the proposed estimation scheme using real-world-based synthesized data with a range-inertial setting. The numerical results demonstrate several advantages of spline embedding in recursive state estimation over the classic discrete-time filtering scheme.

{{</citation>}}


## cs.RO (2)



### (25/44) Effects of Explanation Specificity on Passengers in Autonomous Driving (Daniel Omeiza et al., 2023)

{{<citation>}}

Daniel Omeiza, Raunak Bhattacharyya, Nick Hawes, Marina Jirotka, Lars Kunze. (2023)  
**Effects of Explanation Specificity on Passengers in Autonomous Driving**  

---
Primary Category: cs.RO
Categories: cs-AI, cs-CY, cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.00633v1)  

---


**ABSTRACT**  
The nature of explanations provided by an explainable AI algorithm has been a topic of interest in the explainable AI and human-computer interaction community. In this paper, we investigate the effects of natural language explanations' specificity on passengers in autonomous driving. We extended an existing data-driven tree-based explainer algorithm by adding a rule-based option for explanation generation. We generated auditory natural language explanations with different levels of specificity (abstract and specific) and tested these explanations in a within-subject user study (N=39) using an immersive physical driving simulation setup. Our results showed that both abstract and specific explanations had similar positive effects on passengers' perceived safety and the feeling of anxiety. However, the specific explanations influenced the desire of passengers to takeover driving control from the autonomous vehicle (AV), while the abstract explanations did not. We conclude that natural language auditory explanations are useful for passengers in autonomous driving, and their specificity levels could influence how much in-vehicle participants would wish to be in control of the driving activity.

{{</citation>}}


### (26/44) CQLite: Communication-Efficient Multi-Robot Exploration Using Coverage-biased Distributed Q-Learning (Ehsan Latif et al., 2023)

{{<citation>}}

Ehsan Latif, Ramviyas Parasuraman. (2023)  
**CQLite: Communication-Efficient Multi-Robot Exploration Using Coverage-biased Distributed Q-Learning**  

---
Primary Category: cs.RO
Categories: cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.00500v1)  

---


**ABSTRACT**  
Frontier exploration and reinforcement learning have historically been used to solve the problem of enabling many mobile robots to autonomously and cooperatively explore complex surroundings. These methods need to keep an internal global map for navigation, but they do not take into consideration the high costs of communication and information sharing between robots. This study offers CQLite, a novel distributed Q-learning technique designed to minimize data communication overhead between robots while achieving rapid convergence and thorough coverage in multi-robot exploration. The proposed CQLite method uses ad hoc map merging, and selectively shares updated Q-values at recently identified frontiers to significantly reduce communication costs. The theoretical analysis of CQLite's convergence and efficiency, together with extensive numerical verification on simulated indoor maps utilizing several robots, demonstrates the method's novelty. With over 2x reductions in computation and communication alongside improved mapping performance, CQLite outperformed cutting-edge multi-robot exploration techniques like Rapidly Exploring Random Trees and Deep Reinforcement Learning.

{{</citation>}}


## cs.SE (3)



### (27/44) LLM4CBI: Taming LLMs to Generate Effective Test Programs for Compiler Bug Isolation (Haoxin Tu et al., 2023)

{{<citation>}}

Haoxin Tu, Zhide Zhou, He Jiang, Imam Nur Bani Yusuf, Yuxian Li, Lingxiao Jiang. (2023)  
**LLM4CBI: Taming LLMs to Generate Effective Test Programs for Compiler Bug Isolation**  

---
Primary Category: cs.SE
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.00593v1)  

---


**ABSTRACT**  
Compiler bugs pose a significant threat to safety-critical applications, and promptly and effectively isolating these bugs is crucial for assuring the quality of compilers. However, the limited availability of debugging information on reported bugs complicates the compiler bug isolation task. Existing compiler bug isolation approaches typically convert the problem into a test program mutation problem, but they are still limited by ineffective mutation strategies or high human effort requirements. Drawing inspiration from the recent progress of pre-trained Large Language Models (LLMs), such as ChatGPT, in code generation, we propose a new approach named LLM4CBI to tame LLMs to generate effective test programs for compiler bug isolation. However, using LLMs directly for test program mutation may not yield the desired results due to the challenges associated with formulating precise prompts and selecting specialized prompts. To overcome the challenges, three new components are designed in LLM4CBI. (1) LLM4CBI utilizes a program complexity-guided prompt production component, which leverages data and control flow analysis to identify the most valuable variables and locations in programs for mutation. (2) LLM4CBI employs a memorized prompt selection component, which adopts reinforcement learning to select specialized prompts for mutating test programs continuously. (3) A test program validation component is proposed to select specialized feedback prompts to avoid repeating the same mistakes during the mutation process. Compared with the state-of-the-art approaches (DiWi and RecBi), our evaluation demonstrates the advantages of LLM4CBI: It isolates more bugs, ranging from 13.6% to 90.9% in various settings, than the other approaches. Additionally, we demonstrate that LLM4CBI is extensible, allowing for easy integration with other LLMs.

{{</citation>}}


### (28/44) ChatGPT vs SBST: A Comparative Assessment of Unit Test Suite Generation (Yutian Tang et al., 2023)

{{<citation>}}

Yutian Tang, Zhijie Liu, Zhichao Zhou, Xiapu Luo. (2023)  
**ChatGPT vs SBST: A Comparative Assessment of Unit Test Suite Generation**  

---
Primary Category: cs.SE
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.00588v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) have demonstrated exceptional success in a wide range of general domain tasks, such as question answering and following instructions. Moreover, LLMs have shown potential in various software engineering applications. In this study, we present a systematic comparison of test suites generated by the ChatGPT LLM and the state-of-the-art SBST tool EvoSuite. Our comparison is based on several critical factors, including correctness, readability, code coverage, and bug detection capability. By highlighting the strengths and weaknesses of LLMs (specifically ChatGPT) in generating unit test cases compared to EvoSuite, this work provides valuable insights into the performance of LLMs in solving software engineering problems. Overall, our findings underscore the potential of LLMs in software engineering and pave the way for further research in this area.

{{</citation>}}


### (29/44) Graph Neural Network based Log Anomaly Detection and Explanation (Zhong Li et al., 2023)

{{<citation>}}

Zhong Li, Jiayang Shi, Matthijs van Leeuwen. (2023)  
**Graph Neural Network based Log Anomaly Detection and Explanation**  

---
Primary Category: cs.SE
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: Anomaly Detection, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.00527v1)  

---


**ABSTRACT**  
Event logs are widely used to record the status of high-tech systems, making log anomaly detection important for monitoring those systems. Most existing log anomaly detection methods take a log event count matrix or log event sequences as input, exploiting quantitative and/or sequential relationships between log events to detect anomalies. Unfortunately, only considering quantitative or sequential relationships may result in many false positives and/or false negatives. To alleviate this problem, we propose a graph-based method for unsupervised log anomaly detection, dubbed Logs2Graphs, which first converts event logs into attributed, directed, and weighted graphs, and then leverages graph neural networks to perform graph-level anomaly detection. Specifically, we introduce One-Class Digraph Inception Convolutional Networks, abbreviated as OCDiGCN, a novel graph neural network model for detecting graph-level anomalies in a collection of attributed, directed, and weighted graphs. By coupling the graph representation and anomaly detection steps, OCDiGCN can learn a representation that is especially suited for anomaly detection, resulting in a high detection accuracy. Importantly, for each identified anomaly, we additionally provide a small subset of nodes that play a crucial role in OCDiGCN's prediction as explanations, which can offer valuable cues for subsequent root cause diagnosis. Experiments on five benchmark datasets show that Logs2Graphs performs at least on par state-of-the-art log anomaly detection methods on simple datasets while largely outperforming state-of-the-art log anomaly detection methods on complicated datasets.

{{</citation>}}


## cs.IR (2)



### (30/44) BioCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval (Qiao Jin et al., 2023)

{{<citation>}}

Qiao Jin, Won Kim, Qingyu Chen, Donald C. Comeau, Lana Yeganova, John Wilbur, Zhiyong Lu. (2023)  
**BioCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval**  

---
Primary Category: cs.IR
Categories: cs-AI, cs-CL, cs-IR, cs.IR, q-bio-QM  
Keywords: GPT, Information Retrieval, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.00589v1)  

---


**ABSTRACT**  
Information retrieval (IR) is essential in biomedical knowledge acquisition and clinical decision support. While recent progress has shown that language model encoders perform better semantic retrieval, training such models requires abundant query-article annotations that are difficult to obtain in biomedicine. As a result, most biomedical IR systems only conduct lexical matching. In response, we introduce BioCPT, a first-of-its-kind Contrastively Pre-trained Transformer model for zero-shot biomedical IR. To train BioCPT, we collected an unprecedented scale of 255 million user click logs from PubMed. With such data, we use contrastive learning to train a pair of closely-integrated retriever and re-ranker. Experimental results show that BioCPT sets new state-of-the-art performance on five biomedical IR tasks, outperforming various baselines including much larger models such as GPT-3-sized cpt-text-XL. In addition, BioCPT also generates better biomedical article and sentence representations for semantic evaluations. As such, BioCPT can be readily applied to various real-world biomedical IR tasks. BioCPT API and code are publicly available at https://github.com/ncbi/BioCPT.

{{</citation>}}


### (31/44) GenRec: Large Language Model for Generative Recommendation (Jianchao Ji et al., 2023)

{{<citation>}}

Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, Yongfeng Zhang. (2023)  
**GenRec: Large Language Model for Generative Recommendation**  

---
Primary Category: cs.IR
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2307.00457v2)  

---


**ABSTRACT**  
In recent years, large language models (LLM) have emerged as powerful tools for diverse natural language processing tasks. However, their potential for recommender systems under the generative recommendation paradigm remains relatively unexplored. This paper presents an innovative approach to recommendation systems using large language models (LLMs) based on text data. In this paper, we present a novel LLM for generative recommendation (GenRec) that utilized the expressive power of LLM to directly generate the target item to recommend, rather than calculating ranking score for each candidate item one by one as in traditional discriminative recommendation. GenRec uses LLM's understanding ability to interpret context, learn user preferences, and generate relevant recommendation. Our proposed approach leverages the vast knowledge encoded in large language models to accomplish recommendation tasks. We first we formulate specialized prompts to enhance the ability of LLM to comprehend recommendation tasks. Subsequently, we use these prompts to fine-tune the LLaMA backbone LLM on a dataset of user-item interactions, represented by textual data, to capture user preferences and item characteristics. Our research underscores the potential of LLM-based generative recommendation in revolutionizing the domain of recommendation systems and offers a foundational framework for future explorations in this field. We conduct extensive experiments on benchmark datasets, and the experiments shows that our GenRec has significant better results on large dataset.

{{</citation>}}


## cs.CL (8)



### (32/44) SSP: Self-Supervised Post-training for Conversational Search (Quan Tu et al., 2023)

{{<citation>}}

Quan Tu, Shen Gao, Xiaolong Wu, Zhao Cao, Ji-Rong Wen, Rui Yan. (2023)  
**SSP: Self-Supervised Post-training for Conversational Search**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.00569v1)  

---


**ABSTRACT**  
Conversational search has been regarded as the next-generation search paradigm. Constrained by data scarcity, most existing methods distill the well-trained ad-hoc retriever to the conversational retriever. However, these methods, which usually initialize parameters by query reformulation to discover contextualized dependency, have trouble in understanding the dialogue structure information and struggle with contextual semantic vanishing. In this paper, we propose \fullmodel (\model) which is a new post-training paradigm with three self-supervised tasks to efficiently initialize the conversational search model to enhance the dialogue structure and contextual semantic understanding. Furthermore, the \model can be plugged into most of the existing conversational models to boost their performance. To verify the effectiveness of our proposed method, we apply the conversational encoder post-trained by \model on the conversational search task using two benchmark datasets: CAsT-19 and CAsT-20. Extensive experiments that our \model can boost the performance of several existing conversational search methods. Our source code is available at \url{https://github.com/morecry/SSP}.

{{</citation>}}


### (33/44) TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on the Tensor-Train Decomposition (Mingxue Xu et al., 2023)

{{<citation>}}

Mingxue Xu, Yao Lei Xu, Danilo P. Mandic. (2023)  
**TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on the Tensor-Train Decomposition**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-LG, cs-NA, cs-NE, cs.CL, math-NA  
Keywords: Embedding, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.00526v1)  

---


**ABSTRACT**  
High-dimensional token embeddings underpin Large Language Models (LLMs), as they can capture subtle semantic information and significantly enhance the modelling of complex language patterns. However, the associated high dimensionality also introduces considerable model parameters, and a prohibitively high model storage. To address this issue, this work proposes an approach based on the Tensor-Train Decomposition (TTD), where each token embedding is treated as a Matrix Product State (MPS) that can be efficiently computed in a distributed manner. The experimental results on GPT-2 demonstrate that, through our approach, the embedding layer can be compressed by a factor of up to 38.40 times, and when the compression factor is 3.31 times, even produced a better performance than the original GPT-2 model.

{{</citation>}}


### (34/44) Large Language Models Enable Few-Shot Clustering (Vijay Viswanathan et al., 2023)

{{<citation>}}

Vijay Viswanathan, Kiril Gashteovski, Carolin Lawrence, Tongshuang Wu, Graham Neubig. (2023)  
**Large Language Models Enable Few-Shot Clustering**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2307.00524v1)  

---


**ABSTRACT**  
Unlike traditional unsupervised clustering, semi-supervised clustering allows users to provide meaningful structure to the data, which helps the clustering algorithm to match the user's intent. Existing approaches to semi-supervised clustering require a significant amount of feedback from an expert to improve the clusters. In this paper, we ask whether a large language model can amplify an expert's guidance to enable query-efficient, few-shot semi-supervised text clustering. We show that LLMs are surprisingly effective at improving clustering. We explore three stages where LLMs can be incorporated into clustering: before clustering (improving input features), during clustering (by providing constraints to the clusterer), and after clustering (using LLMs post-correction). We find incorporating LLMs in the first two stages can routinely provide significant improvements in cluster quality, and that LLMs enable a user to make trade-offs between cost and accuracy to produce desired clusters. We release our code and LLM prompts for the public to use.

{{</citation>}}


### (35/44) HeGeL: A Novel Dataset for Geo-Location from Hebrew Text (Tzuf Paz-Argaman et al., 2023)

{{<citation>}}

Tzuf Paz-Argaman, Tal Bauman, Itai Mondshine, Itzhak Omer, Sagi Dalyot, Reut Tsarfaty. (2023)  
**HeGeL: A Novel Dataset for Geo-Location from Hebrew Text**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.00509v1)  

---


**ABSTRACT**  
The task of textual geolocation - retrieving the coordinates of a place based on a free-form language description - calls for not only grounding but also natural language understanding and geospatial reasoning. Even though there are quite a few datasets in English used for geolocation, they are currently based on open-source data (Wikipedia and Twitter), where the location of the described place is mostly implicit, such that the location retrieval resolution is limited. Furthermore, there are no datasets available for addressing the problem of textual geolocation in morphologically rich and resource-poor languages, such as Hebrew. In this paper, we present the Hebrew Geo-Location (HeGeL) corpus, designed to collect literal place descriptions and analyze lingual geospatial reasoning. We crowdsourced 5,649 literal Hebrew place descriptions of various place types in three cities in Israel. Qualitative and empirical analysis show that the data exhibits abundant use of geospatial reasoning and requires a novel environmental representation.

{{</citation>}}


### (36/44) PatternGPT :A Pattern-Driven Framework for Large Language Model Text Generation (Le Xiao et al., 2023)

{{<citation>}}

Le Xiao, Xin Shan. (2023)  
**PatternGPT :A Pattern-Driven Framework for Large Language Model Text Generation**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2307.00470v4)  

---


**ABSTRACT**  
Large language models(LLMS)have shown excellent text generation capabilities, capable of generating fluent human-like responses for many downstream tasks. However, applying large language models to real-world critical tasks remains challenging due to their susceptibility to hallucinations and inability to directly use external knowledge. To cope with the above challenges, this paper proposes PatternGPT, a pattern-driven text generation framework for Large Language Models. Firstly, the framework utilizes the extraction capability of Large Language Models to generate rich and diversified structured and formalized patterns, which facilitates the introduction of external knowledge to do the computation, and then draws on the idea of federated learning to use multiple agents to achieve the sharing in order to obtain more diversified patterns, and finally uses judgment criteria and optimization algorithm to search for high-quality patterns to guide the generation of models. Finally, external knowledge such as judgment criteria and optimization algorithms are used to search for high-quality patterns, and the searched patterns are used to guide model generation. This framework has the advantages of generating diversified patterns, protecting data privacy, combining external knowledge, and improving the quality of generation, which provides an effective method to optimize the text generation capability of large language models, and make it better applied to the field of intelligent dialogue and content generation.

{{</citation>}}


### (37/44) Conformer LLMs -- Convolution Augmented Large Language Models (Prateek Verma, 2023)

{{<citation>}}

Prateek Verma. (2023)  
**Conformer LLMs -- Convolution Augmented Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-LG, cs-MM, cs-SD, cs.CL  
Keywords: Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.00461v1)  

---


**ABSTRACT**  
This work builds together two popular blocks of neural architecture, namely convolutional layers and Transformers, for large language models (LLMs). Non-causal conformers are used ubiquitously in automatic speech recognition. This work aims to adapt these architectures in a causal setup for training LLMs. Transformers decoders effectively capture long-range dependencies over several modalities and form a core backbone of modern advancements in machine learning. Convolutional architectures have been popular in extracting features in domains such as raw 1-D signals, speech, and images, to name a few. In this paper, by combining local and global dependencies over latent representations using causal convolutional filters and Transformer, we achieve significant gains in performance. This work showcases a robust speech architecture that can be integrated and adapted in a causal setup beyond speech applications for large-scale language modeling.

{{</citation>}}


### (38/44) Don't Stop Self-Supervision: Accent Adaptation of Speech Representations via Residual Adapters (Anshu Bhatia et al., 2023)

{{<citation>}}

Anshu Bhatia, Sanchit Sinha, Saket Dingliwal, Karthik Gopalakrishnan, Sravan Bodapati, Katrin Kirchhoff. (2023)  
**Don't Stop Self-Supervision: Accent Adaptation of Speech Representations via Residual Adapters**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.00453v1)  

---


**ABSTRACT**  
Speech representations learned in a self-supervised fashion from massive unlabeled speech corpora have been adapted successfully toward several downstream tasks. However, such representations may be skewed toward canonical data characteristics of such corpora and perform poorly on atypical, non-native accented speaker populations. With the state-of-the-art HuBERT model as a baseline, we propose and investigate self-supervised adaptation of speech representations to such populations in a parameter-efficient way via training accent-specific residual adapters. We experiment with 4 accents and choose automatic speech recognition (ASR) as the downstream task of interest. We obtain strong word error rate reductions (WERR) over HuBERT-large for all 4 accents, with a mean WERR of 22.7% with accent-specific adapters and a mean WERR of 25.1% if the entire encoder is accent-adapted. While our experiments utilize HuBERT and ASR as the downstream task, our proposed approach is both model and task-agnostic.

{{</citation>}}


### (39/44) A Dual-Stream Recurrence-Attention Network with Global-Local Awareness for Emotion Recognition in Textual Dialogue (Jiang Li et al., 2023)

{{<citation>}}

Jiang Li, Xiaoping Wang, Zhigang Zeng. (2023)  
**A Dual-Stream Recurrence-Attention Network with Global-Local Awareness for Emotion Recognition in Textual Dialogue**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Attention, Dialog, Dialogue, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2307.00449v1)  

---


**ABSTRACT**  
In real-world dialogue systems, the ability to understand the user's emotions and interact anthropomorphically is of great significance. Emotion Recognition in Conversation (ERC) is one of the key ways to accomplish this goal and has attracted growing attention. How to model the context in a conversation is a central aspect and a major challenge of ERC tasks. Most existing approaches are generally unable to capture both global and local contextual information efficiently, and their network structures are too complex to design. For this reason, in this work, we propose a straightforward Dual-stream Recurrence-Attention Network (DualRAN) based on Recurrent Neural Network (RNN) and Multi-head ATtention network (MAT). The proposed model eschews the complex network structure of current methods and focuses on combining recurrence-based methods with attention-based methods. DualRAN is a dual-stream structure mainly consisting of local- and global-aware modules, modeling a conversation from distinct perspectives. To achieve the local-aware module, we extend the structure of RNN, thus enhancing the expressive capability of the network. In addition, we develop two single-stream network variants for DualRAN, i.e., SingleRANv1 and SingleRANv2. We conduct extensive experiments on four widely used benchmark datasets, and the results reveal that the proposed model outshines all baselines. Ablation studies further demonstrate the effectiveness of each component.

{{</citation>}}


## cs.IT (1)



### (40/44) Contextual Beamforming: Exploiting Location and AI for Enhanced Wireless Telecommunication Performance (Jaspreet Kaur et al., 2023)

{{<citation>}}

Jaspreet Kaur, Satyam Bhatti, Olaoluwa R Popoola, Muhammad Ali Imran, Rami Ghannam, Qammer H Abbasi, Hasan T Abbas. (2023)  
**Contextual Beamforming: Exploiting Location and AI for Enhanced Wireless Telecommunication Performance**  

---
Primary Category: cs.IT
Categories: cs-IT, cs-SY, cs.IT, eess-SY, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10183v1)  

---


**ABSTRACT**  
The pervasive nature of wireless telecommunication has made it the foundation for mainstream technologies like automation, smart vehicles, virtual reality, and unmanned aerial vehicles. As these technologies experience widespread adoption in our daily lives, ensuring the reliable performance of cellular networks in mobile scenarios has become a paramount challenge. Beamforming, an integral component of modern mobile networks, enables spatial selectivity and improves network quality. However, many beamforming techniques are iterative, introducing unwanted latency to the system. In recent times, there has been a growing interest in leveraging mobile users' location information to expedite beamforming processes. This paper explores the concept of contextual beamforming, discussing its advantages, disadvantages and implications. Notably, the study presents an impressive 53% improvement in signal-to-noise ratio (SNR) by implementing the adaptive beamforming (MRT) algorithm compared to scenarios without beamforming. It further elucidates how MRT contributes to contextual beamforming. The importance of localization in implementing contextual beamforming is also examined. Additionally, the paper delves into the use of artificial intelligence schemes, including machine learning and deep learning, in implementing contextual beamforming techniques that leverage user location information. Based on the comprehensive review, the results suggest that the combination of MRT and Zero forcing (ZF) techniques, alongside deep neural networks (DNN) employing Bayesian Optimization (BO), represents the most promising approach for contextual beamforming. Furthermore, the study discusses the future potential of programmable switches, such as Tofino, in enabling location-aware beamforming.

{{</citation>}}


## eess.IV (2)



### (41/44) ARHNet: Adaptive Region Harmonization for Lesion-aware Augmentation to Improve Segmentation Performance (Jiayu Huo et al., 2023)

{{<citation>}}

Jiayu Huo, Yang Liu, Xi Ouyang, Alejandro Granados, Sebastien Ourselin, Rachel Sparks. (2023)  
**ARHNet: Adaptive Region Harmonization for Lesion-aware Augmentation to Improve Segmentation Performance**  

---
Primary Category: eess.IV
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.01220v1)  

---


**ABSTRACT**  
Accurately segmenting brain lesions in MRI scans is critical for providing patients with prognoses and neurological monitoring. However, the performance of CNN-based segmentation methods is constrained by the limited training set size. Advanced data augmentation is an effective strategy to improve the model's robustness. However, they often introduce intensity disparities between foreground and background areas and boundary artifacts, which weakens the effectiveness of such strategies. In this paper, we propose a foreground harmonization framework (ARHNet) to tackle intensity disparities and make synthetic images look more realistic. In particular, we propose an Adaptive Region Harmonization (ARH) module to dynamically align foreground feature maps to the background with an attention mechanism. We demonstrate the efficacy of our method in improving the segmentation performance using real and synthetic images. Experimental results on the ATLAS 2.0 dataset show that ARHNet outperforms other methods for image harmonization tasks, and boosts the down-stream segmentation performance. Our code is publicly available at https://github.com/King-HAW/ARHNet.

{{</citation>}}


### (42/44) SUGAR: Spherical Ultrafast Graph Attention Framework for Cortical Surface Registration (Jianxun Ren et al., 2023)

{{<citation>}}

Jianxun Ren, Ning An, Youjia Zhang, Danyang Wang, Zhenyu Sun, Cong Lin, Weigang Cui, Weiwei Wang, Ying Zhou, Wei Zhang, Qingyu Hu, Ping Zhang, Dan Hu, Danhong Wang, Hesheng Liu. (2023)  
**SUGAR: Spherical Ultrafast Graph Attention Framework for Cortical Surface Registration**  

---
Primary Category: eess.IV
Categories: cs-CV, cs-LG, eess-IV, eess.IV, q-bio-NC  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.00511v1)  

---


**ABSTRACT**  
Cortical surface registration plays a crucial role in aligning cortical functional and anatomical features across individuals. However, conventional registration algorithms are computationally inefficient. Recently, learning-based registration algorithms have emerged as a promising solution, significantly improving processing efficiency. Nonetheless, there remains a gap in the development of a learning-based method that exceeds the state-of-the-art conventional methods simultaneously in computational efficiency, registration accuracy, and distortion control, despite the theoretically greater representational capabilities of deep learning approaches. To address the challenge, we present SUGAR, a unified unsupervised deep-learning framework for both rigid and non-rigid registration. SUGAR incorporates a U-Net-based spherical graph attention network and leverages the Euler angle representation for deformation. In addition to the similarity loss, we introduce fold and multiple distortion losses, to preserve topology and minimize various types of distortions. Furthermore, we propose a data augmentation strategy specifically tailored for spherical surface registration, enhancing the registration performance. Through extensive evaluation involving over 10,000 scans from 7 diverse datasets, we showed that our framework exhibits comparable or superior registration performance in accuracy, distortion, and test-retest reliability compared to conventional and learning-based methods. Additionally, SUGAR achieves remarkable sub-second processing times, offering a notable speed-up of approximately 12,000 times in registering 9,000 subjects from the UK Biobank dataset in just 32 minutes. This combination of high registration performance and accelerated processing time may greatly benefit large-scale neuroimaging studies.

{{</citation>}}


## q-fin.PR (1)



### (43/44) Pricing European Options with Google AutoML, TensorFlow, and XGBoost (Juan Esteban Berger, 2023)

{{<citation>}}

Juan Esteban Berger. (2023)  
**Pricing European Options with Google AutoML, TensorFlow, and XGBoost**  

---
Primary Category: q-fin.PR
Categories: cs-LG, q-fin-PR, q-fin.PR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2307.00476v1)  

---


**ABSTRACT**  
Researchers have been using Neural Networks and other related machine-learning techniques to price options since the early 1990s. After three decades of improvements in machine learning techniques, computational processing power, cloud computing, and data availability, this paper is able to provide a comparison of using Google Cloud's AutoML Regressor, TensorFlow Neural Networks, and XGBoost Gradient Boosting Decision Trees for pricing European Options. All three types of models were able to outperform the Black Scholes Model in terms of mean absolute error. These results showcase the potential of using historical data from an option's underlying asset for pricing European options, especially when using machine learning algorithms that learn complex patterns that traditional parametric models do not take into account.

{{</citation>}}


## cs.CR (1)



### (44/44) 3D-IDS: Doubly Disentangled Dynamic Intrusion Detection (Chenyang Qiu et al., 2023)

{{<citation>}}

Chenyang Qiu, Yingsheng Geng, Junrui Lu, Kaida Chen, Shitong Zhu, Ya Su, Guoshun Nan, Can Zhang, Junsong Fu, Qimei Cui, Xiaofeng Tao. (2023)  
**3D-IDS: Doubly Disentangled Dynamic Intrusion Detection**  

---
Primary Category: cs.CR
Categories: cs-AI, cs-CR, cs-LG, cs-SI, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2307.11079v1)  

---


**ABSTRACT**  
Network-based intrusion detection system (NIDS) monitors network traffic for malicious activities, forming the frontline defense against increasing attacks over information infrastructures. Although promising, our quantitative analysis shows that existing methods perform inconsistently in declaring various unknown attacks (e.g., 9% and 35% F1 respectively for two distinct unknown threats for an SVM-based method) or detecting diverse known attacks (e.g., 31% F1 for the Backdoor and 93% F1 for DDoS by a GCN-based state-of-the-art method), and reveals that the underlying cause is entangled distributions of flow features. This motivates us to propose 3D-IDS, a novel method that aims to tackle the above issues through two-step feature disentanglements and a dynamic graph diffusion scheme. Specifically, we first disentangle traffic features by a non-parameterized optimization based on mutual information, automatically differentiating tens and hundreds of complex features of various attacks. Such differentiated features will be fed into a memory model to generate representations, which are further disentangled to highlight the attack-specific features. Finally, we use a novel graph diffusion method that dynamically fuses the network topology for spatial-temporal aggregation in evolving data streams. By doing so, we can effectively identify various attacks in encrypted traffics, including unknown threats and known ones that are not easily detected. Experiments show the superiority of our 3D-IDS. We also demonstrate that our two-step feature disentanglements benefit the explainability of NIDS.

{{</citation>}}
