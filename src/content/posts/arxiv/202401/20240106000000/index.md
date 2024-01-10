---
draft: false
title: "arXiv @ 2024.01.06"
date: 2024-01-06
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.06"
    identifier: arxiv_20240106
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.SI (3)](#cssi-3)
- [cs.PL (1)](#cspl-1)
- [cs.SD (4)](#cssd-4)
- [eess.IV (2)](#eessiv-2)
- [q-bio.NC (1)](#q-bionc-1)
- [cs.LG (18)](#cslg-18)
- [cs.AI (1)](#csai-1)
- [cs.HC (3)](#cshc-3)
- [cs.CV (18)](#cscv-18)
- [eess.AS (2)](#eessas-2)
- [cs.CL (22)](#cscl-22)
- [cs.CY (4)](#cscy-4)
- [q-fin.TR (1)](#q-fintr-1)
- [cs.RO (1)](#csro-1)
- [cs.DC (1)](#csdc-1)
- [cs.SE (1)](#csse-1)
- [math.NA (1)](#mathna-1)
- [cs.NE (3)](#csne-3)
- [cs.IR (1)](#csir-1)
- [q-bio.BM (1)](#q-biobm-1)
- [cs.DB (1)](#csdb-1)
- [stat.ML (1)](#statml-1)

## cs.SI (3)



### (1/91) Large Language Models for Social Networks: Applications, Challenges, and Solutions (Jingying Zeng et al., 2024)

{{<citation>}}

Jingying Zeng, Richard Huang, Waleed Malik, Langxuan Yin, Bojan Babic, Danny Shacham, Xiao Yan, Jaewon Yang, Qi He. (2024)  
**Large Language Models for Social Networks: Applications, Challenges, and Solutions**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Language Model, Social Network  
[Paper Link](http://arxiv.org/abs/2401.02575v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are transforming the way people generate, explore, and engage with content. We study how we can develop LLM applications for online social networks. Despite LLMs' successes in other domains, it is challenging to develop LLM-based products for social networks for numerous reasons, and it has been relatively under-reported in the research community. We categorize LLM applications for social networks into three categories. First is knowledge tasks where users want to find new knowledge and information, such as search and question-answering. Second is entertainment tasks where users want to consume interesting content, such as getting entertaining notification content. Third is foundational tasks that need to be done to moderate and operate the social networks, such as content annotation and LLM monitoring. For each task, we share the challenges we found, solutions we developed, and lessons we learned. To the best of our knowledge, this is the first comprehensive paper about developing LLM applications for social networks.

{{</citation>}}


### (2/91) A Community Detection and Graph Neural Network Based Link Prediction Approach for Scientific Literature (Chunjiang Liu et al., 2024)

{{<citation>}}

Chunjiang Liu, Yikun Han, Haiyun Xu, Shihan Yang, Kaidi Wang, Yongye Su. (2024)  
**A Community Detection and Graph Neural Network Based Link Prediction Approach for Scientific Literature**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.02542v1)  

---


**ABSTRACT**  
This study introduces an innovative approach that integrates community detection algorithms with Graph Neural Network (GNN) models to enhance link prediction in scientific literature networks. We specifically focus on the utilization of the Louvain community detection algorithm to uncover latent community structures within these networks, which are then incorporated into GNN architectures to predict potential links. Our methodology demonstrates the importance of understanding community dynamics in complex networks and leverages the strengths of both community detection and GNNs to improve predictive accuracy. Through extensive experiments on bipartite graphs representing scientific collaborations and citations, our approach not only highlights the synergy between community detection and GNNs but also addresses some of the prevalent challenges in link prediction, such as scalability and resolution limits. The results suggest that incorporating community-level information can significantly enhance the performance of GNNs in link prediction tasks. This work contributes to the evolving field of network science by offering a novel perspective on integrating advanced machine learning techniques with traditional network analysis methods to better understand and predict the intricate patterns of scientific collaborations.

{{</citation>}}


### (3/91) Politics and Propaganda on Social Media: How Twitter and Meta Moderate State-Linked Information Operations (Nihat Mugurtay et al., 2024)

{{<citation>}}

Nihat Mugurtay, Umut Duygu, Onur Varol. (2024)  
**Politics and Propaganda on Social Media: How Twitter and Meta Moderate State-Linked Information Operations**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2401.02095v1)  

---


**ABSTRACT**  
Why do Social Media Corporations (SMCs) engage in state-linked information operations? Social media can significantly influence the global political landscape, allowing governments and other political entities to engage in concerted information operations, shaping or manipulating domestic and foreign political agendas. In response to state-linked political manipulation tactics on social media, Twitter and Meta carried out take-down operations against propaganda networks, accusing them of interfering foreign elections, organizing disinformation campaigns, manipulating political debates and many other issues. This research investigates the two SMCs' policy orientation to explain which factors can affect these two companies' reaction against state-linked information operations. We find that good governance indicators such as democracy are significant elements of SMCs' country-focus. This article also examines whether Meta and Twitter's attention to political regime characteristics is influenced by international political alignments. This research illuminates recent trends in SMCs' take-down operations and illuminating interplay between geopolitics and domestic regime characteristics.

{{</citation>}}


## cs.PL (1)



### (4/91) Correct and Compositional Hardware Generators (Rachit Nigam et al., 2024)

{{<citation>}}

Rachit Nigam, Ethan Gabizon, Edmund Lam, Adrian Sampson. (2024)  
**Correct and Compositional Hardware Generators**  

---
Primary Category: cs.PL  
Categories: cs-AR, cs-PL, cs.PL  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.02570v1)  

---


**ABSTRACT**  
Hardware generators help designers explore families of concrete designs and their efficiency trade-offs. Both parameterized hardware description languages (HDLs) and higher-level programming models, however, can obstruct composability. Different concrete designs in a family can have dramatically different timing behavior, and high-level hardware generators rarely expose a consistent HDL-level interface. Composition, therefore, is typically only feasible at the level of individual instances: the user generates concrete designs and then composes them, sacrificing the ability to parameterize the combined design.   We design Parafil, a system for correctly composing hardware generators. Parafil builds on Filament, an HDL with strong compile-time guarantees, and lifts those guarantees to generators to prove that all possible instantiations are free of timing bugs. Parafil can integrate with external hardware generators via a novel system of output parameters and a framework for invoking generator tools. We conduct experiments with two other generators, FloPoCo and Google's XLS, and we implement a parameterized FFT generator to show that Parafil ensures correct design space exploration.

{{</citation>}}


## cs.SD (4)



### (5/91) Siamese Residual Neural Network for Musical Shape Evaluation in Piano Performance Assessment (Xiaoquan Li et al., 2024)

{{<citation>}}

Xiaoquan Li, Stephan Weiss, Yijun Yan, Yinhe Li, Jinchang Ren, John Soraghan, Ming Gong. (2024)  
**Siamese Residual Neural Network for Musical Shape Evaluation in Piano Performance Assessment**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02566v1)  

---


**ABSTRACT**  
Understanding and identifying musical shape plays an important role in music education and performance assessment. To simplify the otherwise time- and cost-intensive musical shape evaluation, in this paper we explore how artificial intelligence (AI) driven models can be applied. Considering musical shape evaluation as a classification problem, a light-weight Siamese residual neural network (S-ResNN) is proposed to automatically identify musical shapes. To assess the proposed approach in the context of piano musical shape evaluation, we have generated a new dataset, containing 4116 music pieces derived by 147 piano preparatory exercises and performed in 28 categories of musical shapes. The experimental results show that the S-ResNN significantly outperforms a number of benchmark methods in terms of the precision, recall and F1 score.

{{</citation>}}


### (6/91) Bridging Modalities: Knowledge Distillation and Masked Training for Translating Multi-Modal Emotion Recognition to Uni-Modal, Speech-Only Emotion Recognition (Muhammad Muaz et al., 2024)

{{<citation>}}

Muhammad Muaz, Nathan Paull, Jahnavi Malagavalli. (2024)  
**Bridging Modalities: Knowledge Distillation and Masked Training for Translating Multi-Modal Emotion Recognition to Uni-Modal, Speech-Only Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.03000v1)  

---


**ABSTRACT**  
This paper presents an innovative approach to address the challenges of translating multi-modal emotion recognition models to a more practical and resource-efficient uni-modal counterpart, specifically focusing on speech-only emotion recognition. Recognizing emotions from speech signals is a critical task with applications in human-computer interaction, affective computing, and mental health assessment. However, existing state-of-the-art models often rely on multi-modal inputs, incorporating information from multiple sources such as facial expressions and gestures, which may not be readily available or feasible in real-world scenarios. To tackle this issue, we propose a novel framework that leverages knowledge distillation and masked training techniques.

{{</citation>}}


### (7/91) An AI-enabled Bias-Free Respiratory Disease Diagnosis Model using Cough Audio: A Case Study for COVID-19 (Tabish Saeed et al., 2024)

{{<citation>}}

Tabish Saeed, Aneeqa Ijaz, Ismail Sadiq, Haneya N. Qureshi, Ali Rizwan, Ali Imran. (2024)  
**An AI-enabled Bias-Free Respiratory Disease Diagnosis Model using Cough Audio: A Case Study for COVID-19**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: AI, Bias, LSTM  
[Paper Link](http://arxiv.org/abs/2401.02996v1)  

---


**ABSTRACT**  
Cough-based diagnosis for Respiratory Diseases (RDs) using Artificial Intelligence (AI) has attracted considerable attention, yet many existing studies overlook confounding variables in their predictive models. These variables can distort the relationship between cough recordings (input data) and RD status (output variable), leading to biased associations and unrealistic model performance. To address this gap, we propose the Bias Free Network (RBFNet), an end to end solution that effectively mitigates the impact of confounders in the training data distribution. RBFNet ensures accurate and unbiased RD diagnosis features, emphasizing its relevance by incorporating a COVID19 dataset in this study. This approach aims to enhance the reliability of AI based RD diagnosis models by navigating the challenges posed by confounding variables. A hybrid of a Convolutional Neural Networks (CNN) and Long-Short Term Memory (LSTM) networks is proposed for the feature encoder module of RBFNet. An additional bias predictor is incorporated in the classification scheme to formulate a conditional Generative Adversarial Network (cGAN) which helps in decorrelating the impact of confounding variables from RD prediction. The merit of RBFNet is demonstrated by comparing classification performance with State of The Art (SoTA) Deep Learning (DL) model (CNN LSTM) after training on different unbalanced COVID-19 data sets, created by using a large scale proprietary cough data set. RBF-Net proved its robustness against extremely biased training scenarios by achieving test set accuracies of 84.1%, 84.6%, and 80.5% for the following confounding variables gender, age, and smoking status, respectively. RBF-Net outperforms the CNN-LSTM model test set accuracies by 5.5%, 7.7%, and 8.2%, respectively

{{</citation>}}


### (8/91) Enhancing Zero-Shot Multi-Speaker TTS with Negated Speaker Representations (Yejin Jeon et al., 2024)

{{<citation>}}

Yejin Jeon, Yunsu Kim, Gary Geunbae Lee. (2024)  
**Enhancing Zero-Shot Multi-Speaker TTS with Negated Speaker Representations**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.02014v1)  

---


**ABSTRACT**  
Zero-shot multi-speaker TTS aims to synthesize speech with the voice of a chosen target speaker without any fine-tuning. Prevailing methods, however, encounter limitations at adapting to new speakers of out-of-domain settings, primarily due to inadequate speaker disentanglement and content leakage. To overcome these constraints, we propose an innovative negation feature learning paradigm that models decoupled speaker attributes as deviations from the complete audio representation by utilizing the subtraction operation. By eliminating superfluous content information from the speaker representation, our negation scheme not only mitigates content leakage, thereby enhancing synthesis robustness, but also improves speaker fidelity. In addition, to facilitate the learning of diverse speaker attributes, we leverage multi-stream Transformers, which retain multiple hypotheses and instigate a training paradigm akin to ensemble learning. To unify these hypotheses and realize the final speaker representation, we employ attention pooling. Finally, in light of the imperative to generate target text utterances in the desired voice, we adopt adaptive layer normalizations to effectively fuse the previously generated speaker representation with the target text representations, as opposed to mere concatenation of the text and audio modalities. Extensive experiments and validations substantiate the efficacy of our proposed approach in preserving and harnessing speaker-specific attributes vis-`a-vis alternative baseline models.

{{</citation>}}


## eess.IV (2)



### (9/91) Vulnerabilities Unveiled: Adversarially Attacking a Multimodal Vision Language Model for Pathology Imaging (Jai Prakash Veerla et al., 2024)

{{<citation>}}

Jai Prakash Veerla, Poojitha Thota, Partha Sai Guttikonda, Shirin Nilizadeh, Jacob M. Luber. (2024)  
**Vulnerabilities Unveiled: Adversarially Attacking a Multimodal Vision Language Model for Pathology Imaging**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, q-bio-TO  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.02565v2)  

---


**ABSTRACT**  
In the dynamic landscape of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted adversarial conditions. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial attacks to intentionally induce misclassifications. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models.

{{</citation>}}


### (10/91) A novel method to enhance pneumonia detection via a model-level ensembling of CNN and vision transformer (Sandeep Angara et al., 2024)

{{<citation>}}

Sandeep Angara, Nishith Reddy Mannuru, Aashrith Mannuru, Sharath Thirunagaru. (2024)  
**A novel method to enhance pneumonia detection via a model-level ensembling of CNN and vision transformer**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2401.02358v1)  

---


**ABSTRACT**  
Pneumonia remains a leading cause of morbidity and mortality worldwide. Chest X-ray (CXR) imaging is a fundamental diagnostic tool, but traditional analysis relies on time-intensive expert evaluation. Recently, deep learning has shown immense potential for automating pneumonia detection from CXRs. This paper explores applying neural networks to improve CXR-based pneumonia diagnosis. We developed a novel model fusing Convolution Neural networks (CNN) and Vision Transformer networks via model-level ensembling. Our fusion architecture combines a ResNet34 variant and a Multi-Axis Vision Transformer small model. Both base models are initialized with ImageNet pre-trained weights. The output layers are removed, and features are combined using a flattening layer before final classification. Experiments used the Kaggle pediatric pneumonia dataset containing 1,341 normal and 3,875 pneumonia CXR images. We compared our model against standalone ResNet34, Vision Transformer, and Swin Transformer Tiny baseline models using identical training procedures. Extensive data augmentation, Adam optimization, learning rate warmup, and decay were employed. The fusion model achieved a state-of-the-art accuracy of 94.87%, surpassing the baselines. We also attained excellent sensitivity, specificity, kappa score, and positive predictive value. Confusion matrix analysis confirms fewer misclassifications. The ResNet34 and Vision Transformer combination enables jointly learning robust features from CNNs and Transformer paradigms. This model-level ensemble technique effectively integrates their complementary strengths for enhanced pneumonia classification.

{{</citation>}}


## q-bio.NC (1)



### (11/91) Memory, Consciousness and Large Language Model (Jitang Li et al., 2024)

{{<citation>}}

Jitang Li, Jinzheng Li. (2024)  
**Memory, Consciousness and Large Language Model**  

---
Primary Category: q-bio.NC  
Categories: cs-AI, cs-CL, q-bio-NC, q-bio.NC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.02509v1)  

---


**ABSTRACT**  
With the development in cognitive science and Large Language Models (LLMs), increasing connections have come to light between these two distinct fields. Building upon these connections, we propose a conjecture suggesting the existence of a duality between LLMs and Tulving's theory of memory. We identify a potential correspondence between Tulving's synergistic ecphory model (SEM) of retrieval and the emergent abilities observed in LLMs, serving as supporting evidence for our conjecture. Furthermore, we speculate that consciousness may be considered a form of emergent ability based on this duality. We also discuss how other theories of consciousness intersect with our research.

{{</citation>}}


## cs.LG (18)



### (12/91) Towards an Adaptable and Generalizable Optimization Engine in Decision and Control: A Meta Reinforcement Learning Approach (Sungwook Yang et al., 2024)

{{<citation>}}

Sungwook Yang, Chaoying Pei, Ran Dai, Chuangchuang Sun. (2024)  
**Towards an Adaptable and Generalizable Optimization Engine in Decision and Control: A Meta Reinforcement Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.02508v1)  

---


**ABSTRACT**  
Sampling-based model predictive control (MPC) has found significant success in optimal control problems with non-smooth system dynamics and cost function. Many machine learning-based works proposed to improve MPC by a) learning or fine-tuning the dynamics/ cost function, or b) learning to optimize for the update of the MPC controllers. For the latter, imitation learning-based optimizers are trained to update the MPC controller by mimicking the expert demonstrations, which, however, are expensive or even unavailable. More significantly, many sequential decision-making problems are in non-stationary environments, requiring that an optimizer should be adaptable and generalizable to update the MPC controller for solving different tasks. To address those issues, we propose to learn an optimizer based on meta-reinforcement learning (RL) to update the controllers. This optimizer does not need expert demonstration and can enable fast adaptation (e.g., few-shots) when it is deployed in unseen control tasks. Experimental results validate the effectiveness of the learned optimizer regarding fast adaptation.

{{</citation>}}


### (13/91) LLM Augmented LLMs: Expanding Capabilities through Composition (Rachit Bansal et al., 2024)

{{<citation>}}

Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Shikhar Vashishth, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, Partha Talukdar. (2024)  
**LLM Augmented LLMs: Expanding Capabilities through Composition**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2401.02412v1)  

---


**ABSTRACT**  
Foundational models with billions of parameters which have been trained on large corpora of data have demonstrated non-trivial skills in a variety of domains. However, due to their monolithic structure, it is challenging and expensive to augment them or impart new skills. On the other hand, due to their adaptation abilities, several new instances of these models are being trained towards new domains and tasks. In this work, we study the problem of efficient and practical composition of existing foundation models with more specific models to enable newer capabilities. To this end, we propose CALM -- Composition to Augment Language Models -- which introduces cross-attention between models to compose their representations and enable new capabilities. Salient features of CALM are: (i) Scales up LLMs on new tasks by 're-using' existing LLMs along with a few additional parameters and data, (ii) Existing model weights are kept intact, and hence preserves existing capabilities, and (iii) Applies to diverse domains and settings. We illustrate that augmenting PaLM2-S with a smaller model trained on low-resource languages results in an absolute improvement of up to 13\% on tasks like translation into English and arithmetic reasoning for low-resource languages. Similarly, when PaLM2-S is augmented with a code-specific model, we see a relative improvement of 40\% over the base model for code generation and explanation tasks -- on-par with fully fine-tuned counterparts.

{{</citation>}}


### (14/91) Real-Time 2D Temperature Field Prediction in Metal Additive Manufacturing Using Physics-Informed Neural Networks (Pouyan Sajadi et al., 2024)

{{<citation>}}

Pouyan Sajadi, Mostafa Rahmani Dehaghani, Yifan Tang, G. Gary Wang. (2024)  
**Real-Time 2D Temperature Field Prediction in Metal Additive Manufacturing Using Physics-Informed Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.02403v1)  

---


**ABSTRACT**  
Accurately predicting the temperature field in metal additive manufacturing (AM) processes is critical to preventing overheating, adjusting process parameters, and ensuring process stability. While physics-based computational models offer precision, they are often time-consuming and unsuitable for real-time predictions and online control in iterative design scenarios. Conversely, machine learning models rely heavily on high-quality datasets, which can be costly and challenging to obtain within the metal AM domain. Our work addresses this by introducing a physics-informed neural network framework specifically designed for temperature field prediction in metal AM. This framework incorporates a physics-informed input, physics-informed loss function, and a Convolutional Long Short-Term Memory (ConvLSTM) architecture. Utilizing real-time temperature data from the process, our model predicts 2D temperature fields for future timestamps across diverse geometries, deposition patterns, and process parameters. We validate the proposed framework in two scenarios: full-field temperature prediction for a thin wall and 2D temperature field prediction for cylinder and cubic parts, demonstrating errors below 3% and 1%, respectively. Our proposed framework exhibits the flexibility to be applied across diverse scenarios with varying process parameters, geometries, and deposition patterns.

{{</citation>}}


### (15/91) A Survey Analyzing Generalization in Deep Reinforcement Learning (Ezgi Korkmaz, 2024)

{{<citation>}}

Ezgi Korkmaz. (2024)  
**A Survey Analyzing Generalization in Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.02349v1)  

---


**ABSTRACT**  
Reinforcement learning research obtained significant success and attention with the utilization of deep neural networks to solve problems in high dimensional state or action spaces. While deep reinforcement learning policies are currently being deployed in many different fields from medical applications to self driving vehicles, there are still ongoing questions the field is trying to answer on the generalization capabilities of deep reinforcement learning policies. In this paper, we will outline the fundamental reasons why deep reinforcement learning policies encounter overfitting problems that limit their robustness and generalization capabilities. Furthermore, we will formalize and unify the diverse solution approaches to increase generalization, and overcome overfitting in state-action value functions. We believe our study can provide a compact systematic unified analysis for the current advancements in deep reinforcement learning, and help to construct robust deep neural policies with improved generalization abilities.

{{</citation>}}


### (16/91) Multi-Source Domain Adaptation with Transformer-based Feature Generation for Subject-Independent EEG-based Emotion Recognition (Shadi Sartipi et al., 2024)

{{<citation>}}

Shadi Sartipi, Mujdat Cetin. (2024)  
**Multi-Source Domain Adaptation with Transformer-based Feature Generation for Subject-Independent EEG-based Emotion Recognition**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Emotion Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2401.02344v1)  

---


**ABSTRACT**  
Although deep learning-based algorithms have demonstrated excellent performance in automated emotion recognition via electroencephalogram (EEG) signals, variations across brain signal patterns of individuals can diminish the model's effectiveness when applied across different subjects. While transfer learning techniques have exhibited promising outcomes, they still encounter challenges related to inadequate feature representations and may overlook the fact that source subjects themselves can possess distinct characteristics. In this work, we propose a multi-source domain adaptation approach with a transformer-based feature generator (MSDA-TF) designed to leverage information from multiple sources. The proposed feature generator retains convolutional layers to capture shallow spatial, temporal, and spectral EEG data representations, while self-attention mechanisms extract global dependencies within these features. During the adaptation process, we group the source subjects based on correlation values and aim to align the moments of the target subject with each source as well as within the sources. MSDA-TF is validated on the SEED dataset and is shown to yield promising results.

{{</citation>}}


### (17/91) Beyond Extraction: Contextualising Tabular Data for Efficient Summarisation by Language Models (Uday Allu et al., 2024)

{{<citation>}}

Uday Allu, Biddwan Ahmed, Vishesh Tripathi. (2024)  
**Beyond Extraction: Contextualising Tabular Data for Efficient Summarisation by Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.02333v1)  

---


**ABSTRACT**  
The conventional use of the Retrieval-Augmented Generation (RAG) architecture has proven effective for retrieving information from diverse documents. However, challenges arise in handling complex table queries, especially within PDF documents containing intricate tabular structures.This research introduces an innovative approach to enhance the accuracy of complex table queries in RAG-based systems. Our methodology involves storing PDFs in the retrieval database and extracting tabular content separately. The extracted tables undergo a process of context enrichment, concatenating headers with corresponding values. To ensure a comprehensive understanding of the enriched data, we employ a fine-tuned version of the Llama-2-chat language model for summarisation within the RAG architecture. Furthermore, we augment the tabular data with contextual sense using the ChatGPT 3.5 API through a one-shot prompt. This enriched data is then fed into the retrieval database alongside other PDFs. Our approach aims to significantly improve the precision of complex table queries, offering a promising solution to a longstanding challenge in information retrieval.

{{</citation>}}


### (18/91) A Robust Quantile Huber Loss With Interpretable Parameter Adjustment In Distributional Reinforcement Learning (Parvin Malekzadeh et al., 2024)

{{<citation>}}

Parvin Malekzadeh, Konstantinos N. Plataniotis, Zissis Poulos, Zeyu Wang. (2024)  
**A Robust Quantile Huber Loss With Interpretable Parameter Adjustment In Distributional Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.02325v2)  

---


**ABSTRACT**  
Distributional Reinforcement Learning (RL) estimates return distribution mainly by learning quantile values via minimizing the quantile Huber loss function, entailing a threshold parameter often selected heuristically or via hyperparameter search, which may not generalize well and can be suboptimal. This paper introduces a generalized quantile Huber loss function derived from Wasserstein distance (WD) calculation between Gaussian distributions, capturing noise in predicted (current) and target (Bellman-updated) quantile values. Compared to the classical quantile Huber loss, this innovative loss function enhances robustness against outliers. Notably, the classical Huber loss function can be seen as an approximation of our proposed loss, enabling parameter adjustment by approximating the amount of noise in the data during the learning process. Empirical tests on Atari games, a common application in distributional RL, and a recent hedging strategy using distributional RL, validate the effectiveness of our proposed loss function and its potential for parameter adjustments in distributional RL. The implementation of the proposed loss function is available here.

{{</citation>}}


### (19/91) Path-based Explanation for Knowledge Graph Completion (Heng Chang et al., 2024)

{{<citation>}}

Heng Chang, Jiangnan Ye, Alejo Lopez Avila, Jinhua Du, Jia Li. (2024)  
**Path-based Explanation for Knowledge Graph Completion**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.02290v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have achieved great success in Knowledge Graph Completion (KGC) by modelling how entities and relations interact in recent years. However, the explanation of the predicted facts has not caught the necessary attention. Proper explanations for the results of GNN-based KGC models increase model transparency and help researchers develop more reliable models. Existing practices for explaining KGC tasks rely on instance/subgraph-based approaches, while in some scenarios, paths can provide more user-friendly and interpretable explanations. Nonetheless, the methods for generating path-based explanations for KGs have not been well-explored. To address this gap, we propose Power-Link, the first path-based KGC explainer that explores GNN-based models. We design a novel simplified graph-powering technique, which enables the generation of path-based explanations with a fully parallelisable and memory-efficient training scheme. We further introduce three new metrics for quantitative evaluation of the explanations, together with a qualitative human evaluation. Extensive experiments demonstrate that Power-Link outperforms the SOTA baselines in interpretability, efficiency, and scalability.

{{</citation>}}


### (20/91) Uncertainty-Aware Deep Attention Recurrent Neural Network for Heterogeneous Time Series Imputation (Linglong Qian et al., 2024)

{{<citation>}}

Linglong Qian, Zina Ibrahim, Richard Dobson. (2024)  
**Uncertainty-Aware Deep Attention Recurrent Neural Network for Heterogeneous Time Series Imputation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Time Series  
[Paper Link](http://arxiv.org/abs/2401.02258v1)  

---


**ABSTRACT**  
Missingness is ubiquitous in multivariate time series and poses an obstacle to reliable downstream analysis. Although recurrent network imputation achieved the SOTA, existing models do not scale to deep architectures that can potentially alleviate issues arising in complex data. Moreover, imputation carries the risk of biased estimations of the ground truth. Yet, confidence in the imputed values is always unmeasured or computed post hoc from model output. We propose DEep Attention Recurrent Imputation (DEARI), which jointly estimates missing values and their associated uncertainty in heterogeneous multivariate time series. By jointly representing feature-wise correlations and temporal dynamics, we adopt a self attention mechanism, along with an effective residual component, to achieve a deep recurrent neural network with good imputation performance and stable convergence. We also leverage self-supervised metric learning to boost performance by optimizing sample similarity. Finally, we transform DEARI into a Bayesian neural network through a novel Bayesian marginalization strategy to produce stochastic DEARI, which outperforms its deterministic equivalent. Experiments show that DEARI surpasses the SOTA in diverse imputation tasks using real-world datasets, namely air quality control, healthcare and traffic.

{{</citation>}}


### (21/91) Policy-regularized Offline Multi-objective Reinforcement Learning (Qian Lin et al., 2024)

{{<citation>}}

Qian Lin, Chao Yu, Zongkai Liu, Zifan Wu. (2024)  
**Policy-regularized Offline Multi-objective Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.02244v1)  

---


**ABSTRACT**  
In this paper, we aim to utilize only offline trajectory data to train a policy for multi-objective RL. We extend the offline policy-regularized method, a widely-adopted approach for single-objective offline RL problems, into the multi-objective setting in order to achieve the above goal. However, such methods face a new challenge in offline MORL settings, namely the preference-inconsistent demonstration problem. We propose two solutions to this problem: 1) filtering out preference-inconsistent demonstrations via approximating behavior preferences, and 2) adopting regularization techniques with high policy expressiveness. Moreover, we integrate the preference-conditioned scalarized update method into policy-regularized offline RL, in order to simultaneously learn a set of policies using a single policy network, thus reducing the computational cost induced by the training of a large number of individual policies for various preferences. Finally, we introduce Regularization Weight Adaptation to dynamically determine appropriate regularization weights for arbitrary target preferences during deployment. Empirical results on various multi-objective datasets demonstrate the capability of our approach in solving offline MORL problems.

{{</citation>}}


### (22/91) U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting (Xiang Ma et al., 2024)

{{<citation>}}

Xiang Ma, Xuemei Li, Lexin Fang, Tianlong Zhao, Caiming Zhang. (2024)  
**U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.02236v1)  

---


**ABSTRACT**  
Time series forecasting is a crucial task in various domains. Caused by factors such as trends, seasonality, or irregular fluctuations, time series often exhibits non-stationary. It obstructs stable feature propagation through deep layers, disrupts feature distributions, and complicates learning data distribution changes. As a result, many existing models struggle to capture the underlying patterns, leading to degraded forecasting performance. In this study, we tackle the challenge of non-stationarity in time series forecasting with our proposed framework called U-Mixer. By combining Unet and Mixer, U-Mixer effectively captures local temporal dependencies between different patches and channels separately to avoid the influence of distribution variations among channels, and merge low- and high-levels features to obtain comprehensive data representations. The key contribution is a novel stationarity correction method, explicitly restoring data distribution by constraining the difference in stationarity between the data before and after model processing to restore the non-stationarity information, while ensuring the temporal dependencies are preserved. Through extensive experiments on various real-world time series datasets, U-Mixer demonstrates its effectiveness and robustness, and achieves 14.5\% and 7.7\% improvements over state-of-the-art (SOTA) methods.

{{</citation>}}


### (23/91) Interpretable Time Series Models for Wastewater Modeling in Combined Sewer Overflows (Teodor Chiaburu et al., 2024)

{{<citation>}}

Teodor Chiaburu, Felix Biessmann. (2024)  
**Interpretable Time Series Models for Wastewater Modeling in Combined Sewer Overflows**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.02465v1)  

---


**ABSTRACT**  
Climate change poses increasingly complex challenges to our society. Extreme weather events such as floods, wild fires or droughts are becoming more frequent, spontaneous and difficult to foresee or counteract. In this work we specifically address the problem of sewage water polluting surface water bodies after spilling over from rain tanks as a consequence of heavy rain events. We investigate to what extent state-of-the-art interpretable time series models can help predict such critical water level points, so that the excess can promptly be redistributed across the sewage network. Our results indicate that modern time series models can contribute to better waste water management and prevention of environmental pollution from sewer systems. All the code and experiments can be found in our repository: https://github.com/TeodorChiaburu/RIWWER_TimeSeries.

{{</citation>}}


### (24/91) Graph Neural Networks for Tabular Data Learning: A Survey with Taxonomy and Directions (Cheng-Te Li et al., 2024)

{{<citation>}}

Cheng-Te Li, Yu-Che Tsai, Chih-Yao Chen, Jay Chiehen Liao. (2024)  
**Graph Neural Networks for Tabular Data Learning: A Survey with Taxonomy and Directions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IR, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.02143v1)  

---


**ABSTRACT**  
In this survey, we dive into Tabular Data Learning (TDL) using Graph Neural Networks (GNNs), a domain where deep learning-based approaches have increasingly shown superior performance in both classification and regression tasks compared to traditional methods. The survey highlights a critical gap in deep neural TDL methods: the underrepresentation of latent correlations among data instances and feature values. GNNs, with their innate capability to model intricate relationships and interactions between diverse elements of tabular data, have garnered significant interest and application across various TDL domains. Our survey provides a systematic review of the methods involved in designing and implementing GNNs for TDL (GNN4TDL). It encompasses a detailed investigation into the foundational aspects and an overview of GNN-based TDL methods, offering insights into their evolving landscape. We present a comprehensive taxonomy focused on constructing graph structures and representation learning within GNN-based TDL methods. In addition, the survey examines various training plans, emphasizing the integration of auxiliary tasks to enhance the effectiveness of instance representations. A critical part of our discussion is dedicated to the practical application of GNNs across a spectrum of GNN4TDL scenarios, demonstrating their versatility and impact. Lastly, we discuss the limitations and propose future research directions, aiming to spur advancements in GNN4TDL. This survey serves as a resource for researchers and practitioners, offering a thorough understanding of GNNs' role in revolutionizing TDL and pointing towards future innovations in this promising area.

{{</citation>}}


### (25/91) Data-Centric Foundation Models in Computational Healthcare: A Survey (Yunkun Zhang et al., 2024)

{{<citation>}}

Yunkun Zhang, Jin Gao, Zheling Tan, Lingfeng Zhou, Kexin Ding, Mu Zhou, Shaoting Zhang, Dequan Wang. (2024)  
**Data-Centric Foundation Models in Computational Healthcare: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02458v1)  

---


**ABSTRACT**  
The advent of foundation models (FMs) as an emerging suite of AI techniques has struck a wave of opportunities in computational healthcare. The interactive nature of these models, guided by pre-training data and human instructions, has ignited a data-centric AI paradigm that emphasizes better data characterization, quality, and scale. In healthcare AI, obtaining and processing high-quality clinical data records has been a longstanding challenge, ranging from data quantity, annotation, patient privacy, and ethics. In this survey, we investigate a wide range of data-centric approaches in the FM era (from model pre-training to inference) towards improving the healthcare workflow. We discuss key perspectives in AI security, assessment, and alignment with human values. Finally, we offer a promising outlook of FM-based analytics to enhance the performance of patient outcome and clinical workflow in the evolving landscape of healthcare and medicine. We provide an up-to-date list of healthcare-related foundation models and datasets at https://github.com/Yunkun-Zhang/Data-Centric-FM-Healthcare .

{{</citation>}}


### (26/91) eCIL-MU: Embedding based Class Incremental Learning and Machine Unlearning (Zhiwei Zuo et al., 2024)

{{<citation>}}

Zhiwei Zuo, Zhuo Tang, Bin Wang, Kenli Li, Anwitaman Datta. (2024)  
**eCIL-MU: Embedding based Class Incremental Learning and Machine Unlearning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.02457v1)  

---


**ABSTRACT**  
New categories may be introduced over time, or existing categories may need to be reclassified. Class incremental learning (CIL) is employed for the gradual acquisition of knowledge about new categories while preserving information about previously learned ones in such dynamic environments. It might also be necessary to also eliminate the influence of related categories on the model to adapt to reclassification. We thus introduce class-level machine unlearning (MU) within CIL. Typically, MU methods tend to be time-consuming and can potentially harm the model's performance. A continuous stream of unlearning requests could lead to catastrophic forgetting. To address these issues, we propose a non-destructive eCIL-MU framework based on embedding techniques to map data into vectors and then be stored in vector databases. Our approach exploits the overlap between CIL and MU tasks for acceleration. Experiments demonstrate the capability of achieving unlearning effectiveness and orders of magnitude (upto $\sim 278\times$) of acceleration.

{{</citation>}}


### (27/91) Re-evaluating the Memory-balanced Pipeline Parallelism: BPipe (Mincong Huang et al., 2024)

{{<citation>}}

Mincong Huang, Chao Wang, Chi Ma, Yineng Zhang, Peng Zhang, Lei Yu. (2024)  
**Re-evaluating the Memory-balanced Pipeline Parallelism: BPipe**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-DC, cs-LG, cs.LG  
Keywords: GPT, LLaMA, Transformer  
[Paper Link](http://arxiv.org/abs/2401.02088v1)  

---


**ABSTRACT**  
Pipeline parallelism is an essential technique in the training of large-scale Transformer models. However, it suffers from imbalanced memory consumption, leading to insufficient memory utilization. The BPipe technique was proposed to address this issue and has proven effective in the GPT-3 model. Nevertheless, our experiments have not yielded similar benefits for LLaMA training. Additionally, BPipe only yields negligible benefits for GPT-3 training when applying flash attention. We analyze the underlying causes of the divergent performance of BPipe on GPT-3 and LLaMA. Furthermore, we introduce a novel method to estimate the performance of BPipe.

{{</citation>}}


### (28/91) View-based Explanations for Graph Neural Networks (Tingyang Chen et al., 2024)

{{<citation>}}

Tingyang Chen, Dazhuo Qiu, Yinghui Wu, Arijit Khan, Xiangyu Ke, Yunjun Gao. (2024)  
**View-based Explanations for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.02086v2)  

---


**ABSTRACT**  
Generating explanations for graph neural networks (GNNs) has been studied to understand their behavior in analytical tasks such as graph classification. Existing approaches aim to understand the overall results of GNNs rather than providing explanations for specific class labels of interest, and may return explanation structures that are hard to access, nor directly queryable.We propose GVEX, a novel paradigm that generates Graph Views for EXplanation. (1) We design a two-tier explanation structure called explanation views. An explanation view consists of a set of graph patterns and a set of induced explanation subgraphs. Given a database G of multiple graphs and a specific class label l assigned by a GNN-based classifier M, it concisely describes the fraction of G that best explains why l is assigned by M. (2) We propose quality measures and formulate an optimization problem to compute optimal explanation views for GNN explanation. We show that the problem is $\Sigma^2_P$-hard. (3) We present two algorithms. The first one follows an explain-and-summarize strategy that first generates high-quality explanation subgraphs which best explain GNNs in terms of feature influence maximization, and then performs a summarization step to generate patterns. We show that this strategy provides an approximation ratio of 1/2. Our second algorithm performs a single-pass to an input node stream in batches to incrementally maintain explanation views, having an anytime quality guarantee of 1/4 approximation. Using real-world benchmark data, we experimentally demonstrate the effectiveness, efficiency, and scalability of GVEX. Through case studies, we showcase the practical applications of GVEX.

{{</citation>}}


### (29/91) A comprehensive survey of research towards AI-enabled unmanned aerial systems in pre-, active-, and post-wildfire management (Sayed Pedram Haeri Boroujeni et al., 2024)

{{<citation>}}

Sayed Pedram Haeri Boroujeni, Abolfazl Razi, Sahand Khoshdel, Fatemeh Afghah, Janice L. Coen, Leo ONeill, Peter Z. Fule, Adam Watts, Nick-Marios T. Kokolakis, Kyriakos G. Vamvoudakis. (2024)  
**A comprehensive survey of research towards AI-enabled unmanned aerial systems in pre-, active-, and post-wildfire management**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.02456v1)  

---


**ABSTRACT**  
Wildfires have emerged as one of the most destructive natural disasters worldwide, causing catastrophic losses in both human lives and forest wildlife. Recently, the use of Artificial Intelligence (AI) in wildfires, propelled by the integration of Unmanned Aerial Vehicles (UAVs) and deep learning models, has created an unprecedented momentum to implement and develop more effective wildfire management. Although some of the existing survey papers have explored various learning-based approaches, a comprehensive review emphasizing the application of AI-enabled UAV systems and their subsequent impact on multi-stage wildfire management is notably lacking. This survey aims to bridge these gaps by offering a systematic review of the recent state-of-the-art technologies, highlighting the advancements of UAV systems and AI models from pre-fire, through the active-fire stage, to post-fire management. To this aim, we provide an extensive analysis of the existing remote sensing systems with a particular focus on the UAV advancements, device specifications, and sensor technologies relevant to wildfire management. We also examine the pre-fire and post-fire management approaches, including fuel monitoring, prevention strategies, as well as evacuation planning, damage assessment, and operation strategies. Additionally, we review and summarize a wide range of computer vision techniques in active-fire management, with an emphasis on Machine Learning (ML), Reinforcement Learning (RL), and Deep Learning (DL) algorithms for wildfire classification, segmentation, detection, and monitoring tasks. Ultimately, we underscore the substantial advancement in wildfire modeling through the integration of cutting-edge AI techniques and UAV-based data, providing novel insights and enhanced predictive capabilities to understand dynamic wildfire behavior.

{{</citation>}}


## cs.AI (1)



### (30/91) On the Prospects of Incorporating Large Language Models (LLMs) in Automated Planning and Scheduling (APS) (Vishal Pallagani et al., 2024)

{{<citation>}}

Vishal Pallagani, Kaushik Roy, Bharath Muppasani, Francesco Fabiano, Andrea Loreggia, Keerthiram Murugesan, Biplav Srivastava, Francesca Rossi, Lior Horesh, Amit Sheth. (2024)  
**On the Prospects of Incorporating Large Language Models (LLMs) in Automated Planning and Scheduling (APS)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.02500v1)  

---


**ABSTRACT**  
Automated Planning and Scheduling is among the growing areas in Artificial Intelligence (AI) where mention of LLMs has gained popularity. Based on a comprehensive review of 126 papers, this paper investigates eight categories based on the unique applications of LLMs in addressing various aspects of planning problems: language translation, plan generation, model construction, multi-agent planning, interactive planning, heuristics optimization, tool integration, and brain-inspired planning. For each category, we articulate the issues considered and existing gaps. A critical insight resulting from our review is that the true potential of LLMs unfolds when they are integrated with traditional symbolic planners, pointing towards a promising neuro-symbolic approach. This approach effectively combines the generative aspects of LLMs with the precision of classical planning methods. By synthesizing insights from existing literature, we underline the potential of this integration to address complex planning challenges. Our goal is to encourage the ICAPS community to recognize the complementary strengths of LLMs and symbolic planners, advocating for a direction in automated planning that leverages these synergistic capabilities to develop more advanced and intelligent planning systems.

{{</citation>}}


## cs.HC (3)



### (31/91) Advancing GUI for Generative AI: Charting the Design Space of Human-AI Interactions through Task Creativity and Complexity (Zijian Ding, 2024)

{{<citation>}}

Zijian Ding. (2024)  
**Advancing GUI for Generative AI: Charting the Design Space of Human-AI Interactions through Task Creativity and Complexity**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.02494v1)  

---


**ABSTRACT**  
Technological progress has persistently shaped the dynamics of human-machine interactions in task execution. In response to the advancements in Generative AI, this paper outlines a detailed study plan that investigates various human-AI interaction modalities across a range of tasks, characterized by differing levels of creativity and complexity. This exploration aims to inform and contribute to the development of Graphical User Interfaces (GUIs) that effectively integrate with and enhance the capabilities of Generative AI systems. The study comprises three parts: exploring fixed-scope tasks through news headline generation, delving into atomic creative tasks with analogy generation, and investigating complex tasks via data visualization. Future work aims to extend this exploration to linearize complex data analysis results into narratives understandable to a broader audience, thereby enhancing the interpretability of AI-generated content.

{{</citation>}}


### (32/91) The Effects of Generative AI on Computing Students' Help-Seeking Preferences (Irene Hou et al., 2024)

{{<citation>}}

Irene Hou, Sophia Metille, Zhuo Li, Owen Man, Cynthia Zastudil, Stephen MacNeil. (2024)  
**The Effects of Generative AI on Computing Students' Help-Seeking Preferences**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.02262v1)  

---


**ABSTRACT**  
Help-seeking is a critical way for students to learn new concepts, acquire new skills, and get unstuck when problem-solving in their computing courses. The recent proliferation of generative AI tools, such as ChatGPT, offers students a new source of help that is always available on-demand. However, it is unclear how this new resource compares to existing help-seeking resources along dimensions of perceived quality, latency, and trustworthiness. In this paper, we investigate the help-seeking preferences and experiences of computing students now that generative AI tools are available to them. We collected survey data (n=47) and conducted interviews (n=8) with computing students. Our results suggest that although these models are being rapidly adopted, they have not yet fully eclipsed traditional help resources. The help-seeking resources that students rely on continue to vary depending on the task and other factors. Finally, we observed preliminary evidence about how help-seeking with generative AI is a skill that needs to be developed, with disproportionate benefits for those who are better able to harness the capabilities of LLMs. We discuss potential implications for integrating generative AI into computing classrooms and the future of help-seeking in the era of generative AI.

{{</citation>}}


### (33/91) MobileAgent: enhancing mobile control via human-machine interaction and SOP integration (Tinghe Ding, 2024)

{{<citation>}}

Tinghe Ding. (2024)  
**MobileAgent: enhancing mobile control via human-machine interaction and SOP integration**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04124v1)  

---


**ABSTRACT**  
Agents centered around Large Language Models (LLMs) are now capable of automating mobile device operations for users. After fine-tuning to learn a user's mobile operations, these agents can adhere to high-level user instructions online. They execute tasks such as goal decomposition, sequencing of sub-goals, and interactive environmental exploration, until the final objective is achieved. However, privacy concerns related to personalized user data arise during mobile operations, requiring user confirmation. Moreover, users' real-world operations are exploratory, with action data being complex and redundant, posing challenges for agent learning. To address these issues, in our practical application, we have designed interactive tasks between agents and humans to identify sensitive information and align with personalized user needs. Additionally, we integrated Standard Operating Procedure (SOP) information within the model's in-context learning to enhance the agent's comprehension of complex task execution. Our approach is evaluated on the new device control benchmark AitW, which encompasses 30K unique instructions across multi-step tasks, including application operation, web searching, and web shopping. Experimental results show that the SOP-based agent achieves state-of-the-art performance without incurring additional inference costs, boasting an overall action success rate of 66.92%.

{{</citation>}}


## cs.CV (18)



### (34/91) Learning to Prompt with Text Only Supervision for Vision-Language Models (Muhammad Uzair Khattak et al., 2024)

{{<citation>}}

Muhammad Uzair Khattak, Muhammad Ferjad Naeem, Muzammal Naseer, Luc Van Gool, Federico Tombari. (2024)  
**Learning to Prompt with Text Only Supervision for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.02418v1)  

---


**ABSTRACT**  
Foundational vision-language models such as CLIP are becoming a new paradigm in vision, due to their excellent generalization abilities. However, adapting these models for downstream tasks while maintaining their generalization remains a challenge. In literature, one branch of methods adapts CLIP by learning prompts using visual information. While effective, most of these works require labeled data which is not practical, and often struggle to generalize towards new datasets due to over-fitting on the source data. An alternative approach resorts to training-free methods by generating class descriptions from large language models (LLMs) and perform prompt ensembling. However, these methods often generate class specific prompts that cannot be transferred to other classes, which incur higher costs by generating LLM descriptions for each class separately. In this work, we propose to combine the strengths of these both streams of methods by learning prompts using only text data derived from LLMs. As supervised training of prompts is not trivial due to absence of images, we develop a training approach that allows prompts to extract rich contextual knowledge from LLM data. Moreover, with LLM contextual data mapped within the learned prompts, it enables zero-shot transfer of prompts to new classes and datasets potentially cutting the LLM prompt engineering cost. To the best of our knowledge, this is the first work that learns generalized prompts using text only data. We perform extensive evaluations on 4 benchmarks where our method improves over prior ensembling works while being competitive to those utilizing labeled images. Our code and pre-trained models are available at https://github.com/muzairkhattak/ProText.

{{</citation>}}


### (35/91) ODIN: A Single Model for 2D and 3D Perception (Ayush Jain et al., 2024)

{{<citation>}}

Ayush Jain, Pushkal Katara, Nikolaos Gkanatsios, Adam W. Harley, Gabriel Sarch, Kriti Aggarwal, Vishrav Chaudhary, Katerina Fragkiadaki. (2024)  
**ODIN: A Single Model for 2D and 3D Perception**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02416v1)  

---


**ABSTRACT**  
State-of-the-art models on contemporary 3D perception benchmarks like ScanNet consume and label dataset-provided 3D point clouds, obtained through post processing of sensed multiview RGB-D images. They are typically trained in-domain, forego large-scale 2D pre-training and outperform alternatives that featurize the posed RGB-D multiview images instead. The gap in performance between methods that consume posed images versus post-processed 3D point clouds has fueled the belief that 2D and 3D perception require distinct model architectures. In this paper, we challenge this view and propose ODIN (Omni-Dimensional INstance segmentation), a model that can segment and label both 2D RGB images and 3D point clouds, using a transformer architecture that alternates between 2D within-view and 3D cross-view information fusion. Our model differentiates 2D and 3D feature operations through the positional encodings of the tokens involved, which capture pixel coordinates for 2D patch tokens and 3D coordinates for 3D feature tokens. ODIN achieves state-of-the-art performance on ScanNet200, Matterport3D and AI2THOR 3D instance segmentation benchmarks, and competitive performance on ScanNet, S3DIS and COCO. It outperforms all previous works by a wide margin when the sensed 3D point cloud is used in place of the point cloud sampled from 3D mesh. When used as the 3D perception engine in an instructable embodied agent architecture, it sets a new state-of-the-art on the TEACh action-from-dialogue benchmark. Our code and checkpoints can be found at the project website: https://odin-seg.github.io.

{{</citation>}}


### (36/91) ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning (Fanqing Meng et al., 2024)

{{<citation>}}

Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, Ping Luo. (2024)  
**ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.02384v1)  

---


**ABSTRACT**  
Charts play a vital role in data visualization, understanding data patterns, and informed decision-making. However, their unique combination of graphical elements (e.g., bars, lines) and textual components (e.g., labels, legends) poses challenges for general-purpose multimodal models. While vision-language models trained on chart data excel in comprehension, they struggle with generalization and require task-specific fine-tuning. To address these challenges, we propose ChartAssistant, a chart-based vision-language model for universal chart comprehension and reasoning. ChartAssistant leverages ChartSFT, a comprehensive dataset covering diverse chart-related tasks with basic and specialized chart types. It undergoes a two-stage training process, starting with pre-training on chart-to-table parsing to align chart and text, followed by multitask instruction-following fine-tuning. This approach enables ChartAssistant to achieve competitive performance across various chart tasks without task-specific fine-tuning. Experimental results demonstrate significant performance gains over the state-of-the-art UniChart method, outperforming OpenAI's GPT-4V(ision) on real-world chart data. The code and data are available at https://github.com/OpenGVLab/ChartAst.

{{</citation>}}


### (37/91) Mining Fine-Grained Image-Text Alignment for Zero-Shot Captioning via Text-Only Training (Longtian Qiu et al., 2024)

{{<citation>}}

Longtian Qiu, Shan Ning, Xuming He. (2024)  
**Mining Fine-Grained Image-Text Alignment for Zero-Shot Captioning via Text-Only Training**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.02347v1)  

---


**ABSTRACT**  
Image captioning aims at generating descriptive and meaningful textual descriptions of images, enabling a broad range of vision-language applications. Prior works have demonstrated that harnessing the power of Contrastive Image Language Pre-training (CLIP) offers a promising approach to achieving zero-shot captioning, eliminating the need for expensive caption annotations. However, the widely observed modality gap in the latent space of CLIP harms the performance of zero-shot captioning by breaking the alignment between paired image-text features. To address this issue, we conduct an analysis on the CLIP latent space which leads to two findings. Firstly, we observe that the CLIP's visual feature of image subregions can achieve closer proximity to the paired caption due to the inherent information loss in text descriptions. In addition, we show that the modality gap between a paired image-text can be empirically modeled as a zero-mean Gaussian distribution. Motivated by the findings, we propose a novel zero-shot image captioning framework with text-only training to reduce the modality gap. In particular, we introduce a subregion feature aggregation to leverage local region information, which produces a compact visual representation for matching text representation. Moreover, we incorporate a noise injection and CLIP reranking strategy to boost captioning performance. We also extend our framework to build a zero-shot VQA pipeline, demonstrating its generality. Through extensive experiments on common captioning and VQA datasets such as MSCOCO, Flickr30k and VQAV2, we show that our method achieves remarkable performance improvements. Code is available at https://github.com/Artanic30/MacCap.

{{</citation>}}


### (38/91) LLaVA-$φ$: Efficient Multi-Modal Assistant with Small Language Model (Yichen Zhu et al., 2024)

{{<citation>}}

Yichen Zhu, Minjie Zhu, Ning Liu, Zhicai Ou, Xiaofeng Mou, Jian Tang. (2024)  
**LLaVA-$φ$: Efficient Multi-Modal Assistant with Small Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.02330v1)  

---


**ABSTRACT**  
In this paper, we introduce LLaVA-$\phi$ (LLaVA-Phi), an efficient multi-modal assistant that harnesses the power of the recently advanced small language model, Phi-2, to facilitate multi-modal dialogues. LLaVA-Phi marks a notable advancement in the realm of compact multi-modal models. It demonstrates that even smaller language models, with as few as 2.7B parameters, can effectively engage in intricate dialogues that integrate both textual and visual elements, provided they are trained with high-quality corpora. Our model delivers commendable performance on publicly available benchmarks that encompass visual comprehension, reasoning, and knowledge-based perception. Beyond its remarkable performance in multi-modal dialogue tasks, our model opens new avenues for applications in time-sensitive environments and systems that require real-time interaction, such as embodied agents. It highlights the potential of smaller language models to achieve sophisticated levels of understanding and interaction, while maintaining greater resource efficiency.The project is available at {https://github.com/zhuyiche/llava-phi}.

{{</citation>}}


### (39/91) ClassWise-SAM-Adapter: Parameter Efficient Fine-tuning Adapts Segment Anything to SAR Domain for Semantic Segmentation (Xinyang Pu et al., 2024)

{{<citation>}}

Xinyang Pu, Hecheng Jia, Linghao Zheng, Feng Wang, Feng Xu. (2024)  
**ClassWise-SAM-Adapter: Parameter Efficient Fine-tuning Adapts Segment Anything to SAR Domain for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2401.02326v1)  

---


**ABSTRACT**  
In the realm of artificial intelligence, the emergence of foundation models, backed by high computing capabilities and extensive data, has been revolutionary. Segment Anything Model (SAM), built on the Vision Transformer (ViT) model with millions of parameters and vast training dataset SA-1B, excels in various segmentation scenarios relying on its significance of semantic information and generalization ability. Such achievement of visual foundation model stimulates continuous researches on specific downstream tasks in computer vision. The ClassWise-SAM-Adapter (CWSAM) is designed to adapt the high-performing SAM for landcover classification on space-borne Synthetic Aperture Radar (SAR) images. The proposed CWSAM freezes most of SAM's parameters and incorporates lightweight adapters for parameter efficient fine-tuning, and a classwise mask decoder is designed to achieve semantic segmentation task. This adapt-tuning method allows for efficient landcover classification of SAR images, balancing the accuracy with computational demand. In addition, the task specific input module injects low frequency information of SAR images by MLP-based layers to improve the model performance. Compared to conventional state-of-the-art semantic segmentation algorithms by extensive experiments, CWSAM showcases enhanced performance with fewer computing resources, highlighting the potential of leveraging foundational models like SAM for specific downstream tasks in the SAR domain. The source code is available at: https://github.com/xypu98/CWSAM.

{{</citation>}}


### (40/91) BA-SAM: Scalable Bias-Mode Attention Mask for Segment Anything Model (Yiran Song et al., 2024)

{{<citation>}}

Yiran Song, Qianyu Zhou, Xiangtai Li, Deng-Ping Fan, Xuequan Lu, Lizhuang Ma. (2024)  
**BA-SAM: Scalable Bias-Mode Attention Mask for Segment Anything Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Bias  
[Paper Link](http://arxiv.org/abs/2401.02317v2)  

---


**ABSTRACT**  
In this paper, we address the challenge of image resolution variation for the Segment Anything Model (SAM). SAM, known for its zero-shot generalizability, exhibits a performance degradation when faced with datasets with varying image sizes. Previous approaches tend to resize the image to a fixed size or adopt structure modifications, hindering the preservation of SAM's rich prior knowledge. Besides, such task-specific tuning necessitates a complete retraining of the model, which is cost-expensive and unacceptable for deployment in the downstream tasks. In this paper, we reformulate this issue as a length extrapolation problem, where token sequence length varies while maintaining a consistent patch size for images of different sizes. To this end, we propose Scalable Bias-Mode Attention Mask (BA-SAM) to enhance SAM's adaptability to varying image resolutions while eliminating the need for structure modifications. Firstly, we introduce a new scaling factor to ensure consistent magnitude in the attention layer's dot product values when the token sequence length changes. Secondly, we present a bias-mode attention mask that allows each token to prioritize neighboring information, mitigating the impact of untrained distant information. Our BA-SAM demonstrates efficacy in two scenarios: zero-shot and fine-tuning. Extensive evaluation on diverse datasets, including DIS5K, DUTS, ISIC, COD10K, and COCO, reveals its ability to significantly mitigate performance degradation in the zero-shot setting and achieve state-of-the-art performance with minimal fine-tuning. Furthermore, we propose a generalized model and benchmark, showcasing BA-SAM's generalizability across all four datasets simultaneously.

{{</citation>}}


### (41/91) SuperEdge: Towards a Generalization Model for Self-Supervised Edge Detection (Leng Kai et al., 2024)

{{<citation>}}

Leng Kai, Zhang Zhijie, Liu Jie, Zed Boukhers, Sui Wei, Cong Yang, Li Zhijun. (2024)  
**SuperEdge: Towards a Generalization Model for Self-Supervised Edge Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.02313v1)  

---


**ABSTRACT**  
Edge detection is a fundamental technique in various computer vision tasks. Edges are indeed effectively delineated by pixel discontinuity and can offer reliable structural information even in textureless areas. State-of-the-art heavily relies on pixel-wise annotations, which are labor-intensive and subject to inconsistencies when acquired manually. In this work, we propose a novel self-supervised approach for edge detection that employs a multi-level, multi-homography technique to transfer annotations from synthetic to real-world datasets. To fully leverage the generated edge annotations, we developed SuperEdge, a streamlined yet efficient model capable of concurrently extracting edges at pixel-level and object-level granularity. Thanks to self-supervised training, our method eliminates the dependency on manual annotated edge labels, thereby enhancing its generalizability across diverse datasets. Comparative evaluations reveal that SuperEdge advances edge detection, demonstrating improvements of 4.9% in ODS and 3.3% in OIS over the existing STEdge method on BIPEDv2.

{{</citation>}}


### (42/91) TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection (Hao Sun et al., 2024)

{{<citation>}}

Hao Sun, Mingyao Zhou, Wenjing Chen, Wei Xie. (2024)  
**TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.02309v2)  

---


**ABSTRACT**  
Video moment retrieval (MR) and highlight detection (HD) based on natural language queries are two highly related tasks, which aim to obtain relevant moments within videos and highlight scores of each video clip. Recently, several methods have been devoted to building DETR-based networks to solve both MR and HD jointly. These methods simply add two separate task heads after multi-modal feature extraction and feature interaction, achieving good performance. Nevertheless, these approaches underutilize the reciprocal relationship between two tasks. In this paper, we propose a task-reciprocal transformer based on DETR (TR-DETR) that focuses on exploring the inherent reciprocity between MR and HD. Specifically, a local-global multi-modal alignment module is first built to align features from diverse modalities into a shared latent space. Subsequently, a visual feature refinement is designed to eliminate query-irrelevant information from visual features for modal interaction. Finally, a task cooperation module is constructed to refine the retrieval pipeline and the highlight score prediction process by utilizing the reciprocity between MR and HD. Comprehensive experiments on QVHighlights, Charades-STA and TVSum datasets demonstrate that TR-DETR outperforms existing state-of-the-art methods. Codes are available at \url{https://github.com/mingyao1120/TR-DETR}.

{{</citation>}}


### (43/91) GridFormer: Point-Grid Transformer for Surface Reconstruction (Shengtao Li et al., 2024)

{{<citation>}}

Shengtao Li, Ge Gao, Yudong Liu, Yu-Shen Liu, Ming Gu. (2024)  
**GridFormer: Point-Grid Transformer for Surface Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.02292v1)  

---


**ABSTRACT**  
Implicit neural networks have emerged as a crucial technology in 3D surface reconstruction. To reconstruct continuous surfaces from discrete point clouds, encoding the input points into regular grid features (plane or volume) has been commonly employed in existing approaches. However, these methods typically use the grid as an index for uniformly scattering point features. Compared with the irregular point features, the regular grid features may sacrifice some reconstruction details but improve efficiency. To take full advantage of these two types of features, we introduce a novel and high-efficiency attention mechanism between the grid and point features named Point-Grid Transformer (GridFormer). This mechanism treats the grid as a transfer point connecting the space and point cloud. Our method maximizes the spatial expressiveness of grid features and maintains computational efficiency. Furthermore, optimizing predictions over the entire space could potentially result in blurred boundaries. To address this issue, we further propose a boundary optimization strategy incorporating margin binary cross-entropy loss and boundary sampling. This approach enables us to achieve a more precise representation of the object structure. Our experiments validate that our method is effective and outperforms the state-of-the-art approaches under widely used benchmarks by producing more precise geometry reconstructions. The code is available at https://github.com/list17/GridFormer.

{{</citation>}}


### (44/91) Distillation-based fabric anomaly detection (Simon Thomine et al., 2024)

{{<citation>}}

Simon Thomine, Hichem Snoussi. (2024)  
**Distillation-based fabric anomaly detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02287v1)  

---


**ABSTRACT**  
Unsupervised texture anomaly detection has been a concerning topic in a vast amount of industrial processes. Patterned textures inspection, particularly in the context of fabric defect detection, is indeed a widely encountered use case. This task involves handling a diverse spectrum of colors and textile types, encompassing a wide range of fabrics. Given the extensive variability in colors, textures, and defect types, fabric defect detection poses a complex and challenging problem in the field of patterned textures inspection. In this article, we propose a knowledge distillation-based approach tailored specifically for addressing the challenge of unsupervised anomaly detection in textures resembling fabrics. Our method aims to redefine the recently introduced reverse distillation approach, which advocates for an encoder-decoder design to mitigate classifier bias and to prevent the student from reconstructing anomalies. In this study, we present a new reverse distillation technique for the specific task of fabric defect detection. Our approach involves a meticulous design selection that strategically highlights high-level features. To demonstrate the capabilities of our approach both in terms of performance and inference speed, we conducted a series of experiments on multiple texture datasets, including MVTEC AD, AITEX, and TILDA, alongside conducting experiments on a dataset acquired from a textile manufacturing facility. The main contributions of this paper are the following: a robust texture anomaly detector utilizing a reverse knowledge-distillation technique suitable for both anomaly detection and domain generalization and a novel dataset encompassing a diverse range of fabrics and defects.

{{</citation>}}


### (45/91) ShapeAug: Occlusion Augmentation for Event Camera Data (Katharina Bendig et al., 2024)

{{<citation>}}

Katharina Bendig, René Schuster, Didier Stricker. (2024)  
**ShapeAug: Occlusion Augmentation for Event Camera Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.02274v1)  

---


**ABSTRACT**  
Recently, Dynamic Vision Sensors (DVSs) sparked a lot of interest due to their inherent advantages over conventional RGB cameras. These advantages include a low latency, a high dynamic range and a low energy consumption. Nevertheless, the processing of DVS data using Deep Learning (DL) methods remains a challenge, particularly since the availability of event training data is still limited. This leads to a need for event data augmentation techniques in order to improve accuracy as well as to avoid over-fitting on the training data. Another challenge especially in real world automotive applications is occlusion, meaning one object is hindering the view onto the object behind it. In this paper, we present a novel event data augmentation approach, which addresses this problem by introducing synthetic events for randomly moving objects in a scene. We test our method on multiple DVS classification datasets, resulting in an relative improvement of up to 6.5 % in top1-accuracy. Moreover, we apply our augmentation technique on the real world Gen1 Automotive Event Dataset for object detection, where we especially improve the detection of pedestrians by up to 5 %.

{{</citation>}}


### (46/91) Marginal Debiased Network for Fair Visual Recognition (Mei Wang et al., 2024)

{{<citation>}}

Mei Wang, Weihong Deng, Sen Su. (2024)  
**Marginal Debiased Network for Fair Visual Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.02150v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) are often prone to learn the spurious correlations between target classes and bias attributes, like gender and race, inherent in a major portion of training data (bias-aligned samples), thus showing unfair behavior and arising controversy in the modern pluralistic and egalitarian society. In this paper, we propose a novel marginal debiased network (MDN) to learn debiased representations. More specifically, a marginal softmax loss (MSL) is designed by introducing the idea of margin penalty into the fairness problem, which assigns a larger margin for bias-conflicting samples (data without spurious correlations) than for bias-aligned ones, so as to deemphasize the spurious correlations and improve generalization on unbiased test criteria. To determine the margins, our MDN is optimized through a meta learning framework. We propose a meta equalized loss (MEL) to perceive the model fairness, and adaptively update the margin parameters by metaoptimization which requires the trained model guided by the optimal margins should minimize MEL computed on an unbiased meta-validation set. Extensive experiments on BiasedMNIST, Corrupted CIFAR-10, CelebA and UTK-Face datasets demonstrate that our MDN can achieve a remarkable performance on under-represented samples and obtain superior debiased results against the previous approaches.

{{</citation>}}


### (47/91) Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions (Oindrila Saha et al., 2024)

{{<citation>}}

Oindrila Saha, Grant Van Horn, Subhransu Maji. (2024)  
**Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.02460v1)  

---


**ABSTRACT**  
The zero-shot performance of existing vision-language models (VLMs) such as CLIP is limited by the availability of large-scale, aligned image and text datasets in specific domains. In this work, we leverage two complementary sources of information -- descriptions of categories generated by large language models (LLMs) and abundant, fine-grained image classification datasets -- to improve the zero-shot classification performance of VLMs across fine-grained domains. On the technical side, we develop methods to train VLMs with this "bag-level" image-text supervision. We find that simply using these attributes at test-time does not improve performance, but our training strategy, for example, on the iNaturalist dataset, leads to an average improvement of 4-5% in zero-shot classification accuracy for novel categories of birds and flowers. Similar improvements are observed in domains where a subset of the categories was used to fine-tune the model. By prompting LLMs in various ways, we generate descriptions that capture visual appearance, habitat, and geographic regions and pair them with existing attributes such as the taxonomic structure of the categories. We systematically evaluate their ability to improve zero-shot categorization in natural domains. Our findings suggest that geographic priors can be just as effective and are complementary to visual appearance. Our method also outperforms prior work on prompt-based tuning of VLMs. We plan to release the benchmark, consisting of 7 datasets, which will contribute to future research in zero-shot recognition.

{{</citation>}}


### (48/91) Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images under Image Degradation (Fahim Faisal Niloy et al., 2024)

{{<citation>}}

Fahim Faisal Niloy, Kishor Kumar Bhaumik, Simon S. Woo. (2024)  
**Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images under Image Degradation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.02113v1)  

---


**ABSTRACT**  
Online adaptation to distribution shifts in satellite image segmentation stands as a crucial yet underexplored problem. In this paper, we address source-free and online domain adaptation, i.e., test-time adaptation (TTA), for satellite images, with the focus on mitigating distribution shifts caused by various forms of image degradation. Towards achieving this goal, we propose a novel TTA approach involving two effective strategies. First, we progressively estimate the global Batch Normalization (BN) statistics of the target distribution with incoming data stream. Leveraging these statistics during inference has the ability to effectively reduce domain gap. Furthermore, we enhance prediction quality by refining the predicted masks using global class centers. Both strategies employ dynamic momentum for fast and stable convergence. Notably, our method is backpropagation-free and hence fast and lightweight, making it highly suitable for on-the-fly adaptation to new domain. Through comprehensive experiments across various domain adaptation scenarios, we demonstrate the robust performance of our method.

{{</citation>}}


### (49/91) Federated Class-Incremental Learning with Prototype Guided Transformer (Haiyang Guo et al., 2024)

{{<citation>}}

Haiyang Guo, Fei Zhu, Wenzhuo Liu, Xu-Yao Zhang, Cheng-Lin Liu. (2024)  
**Federated Class-Incremental Learning with Prototype Guided Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.02094v1)  

---


**ABSTRACT**  
Existing federated learning methods have effectively addressed decentralized learning in scenarios involving data privacy and non-IID data. However, in real-world situations, each client dynamically learns new classes, requiring the global model to maintain discriminative capabilities for both new and old classes. To effectively mitigate the effects of catastrophic forgetting and data heterogeneity under low communication costs, we designed a simple and effective method named PLoRA. On the one hand, we adopt prototype learning to learn better feature representations and leverage the heuristic information between prototypes and class features to design a prototype re-weight module to solve the classifier bias caused by data heterogeneity without retraining the classification layer. On the other hand, our approach utilizes a pre-trained model as the backbone and utilizes LoRA to fine-tune with a tiny amount of parameters when learning new classes. Moreover, PLoRA does not rely on similarity-based module selection strategies, thereby further reducing communication overhead. Experimental results on standard datasets indicate that our method outperforms the state-of-the-art approaches significantly. More importantly, our method exhibits strong robustness and superiority in various scenarios and degrees of data heterogeneity. Our code will be publicly available.

{{</citation>}}


### (50/91) DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection (Yunfan Ye et al., 2024)

{{<citation>}}

Yunfan Ye, Kai Xu, Yuhang Huang, Renjiao Yi, Zhiping Cai. (2024)  
**DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02032v2)  

---


**ABSTRACT**  
Limited by the encoder-decoder architecture, learning-based edge detectors usually have difficulty predicting edge maps that satisfy both correctness and crispness. With the recent success of the diffusion probabilistic model (DPM), we found it is especially suitable for accurate and crisp edge detection since the denoising process is directly applied to the original image size. Therefore, we propose the first diffusion model for the task of general edge detection, which we call DiffusionEdge. To avoid expensive computational resources while retaining the final performance, we apply DPM in the latent space and enable the classic cross-entropy loss which is uncertainty-aware in pixel level to directly optimize the parameters in latent space in a distillation manner. We also adopt a decoupled architecture to speed up the denoising process and propose a corresponding adaptive Fourier filter to adjust the latent features of specific frequencies. With all the technical designs, DiffusionEdge can be stably trained with limited resources, predicting crisp and accurate edge maps with much fewer augmentation strategies. Extensive experiments on four edge detection benchmarks demonstrate the superiority of DiffusionEdge both in correctness and crispness. On the NYUDv2 dataset, compared to the second best, we increase the ODS, OIS (without post-processing) and AC by 30.2%, 28.1% and 65.1%, respectively. Code: https://github.com/GuHuangAI/DiffusionEdge.

{{</citation>}}


### (51/91) Spy-Watermark: Robust Invisible Watermarking for Backdoor Attack (Ruofei Wang et al., 2024)

{{<citation>}}

Ruofei Wang, Renjie Wan, Zongyu Guo, Qing Guo, Rui Huang. (2024)  
**Spy-Watermark: Robust Invisible Watermarking for Backdoor Attack**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.02031v1)  

---


**ABSTRACT**  
Backdoor attack aims to deceive a victim model when facing backdoor instances while maintaining its performance on benign data. Current methods use manual patterns or special perturbations as triggers, while they often overlook the robustness against data corruption, making backdoor attacks easy to defend in practice. To address this issue, we propose a novel backdoor attack method named Spy-Watermark, which remains effective when facing data collapse and backdoor defense. Therein, we introduce a learnable watermark embedded in the latent domain of images, serving as the trigger. Then, we search for a watermark that can withstand collapse during image decoding, cooperating with several anti-collapse operations to further enhance the resilience of our trigger against data corruption. Extensive experiments are conducted on CIFAR10, GTSRB, and ImageNet datasets, demonstrating that Spy-Watermark overtakes ten state-of-the-art methods in terms of robustness and stealthiness.

{{</citation>}}


## eess.AS (2)



### (52/91) Task Oriented Dialogue as a Catalyst for Self-Supervised Automatic Speech Recognition (David M. Chan et al., 2024)

{{<citation>}}

David M. Chan, Shalini Ghosh, Hitesh Tulsiani, Ariya Rastrow, Björn Hoffmeister. (2024)  
**Task Oriented Dialogue as a Catalyst for Self-Supervised Automatic Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Contrastive Learning, Dialog, Dialogue, NLU, Self-Supervised, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.02417v1)  

---


**ABSTRACT**  
While word error rates of automatic speech recognition (ASR) systems have consistently fallen, natural language understanding (NLU) applications built on top of ASR systems still attribute significant numbers of failures to low-quality speech recognition results. Existing assistant systems collect large numbers of these unsuccessful interactions, but these systems usually fail to learn from these interactions, even in an offline fashion. In this work, we introduce CLC: Contrastive Learning for Conversations, a family of methods for contrastive fine-tuning of models in a self-supervised fashion, making use of easily detectable artifacts in unsuccessful conversations with assistants. We demonstrate that our CLC family of approaches can improve the performance of ASR models on OD3, a new public large-scale semi-synthetic meta-dataset of audio task-oriented dialogues, by up to 19.2%. These gains transfer to real-world systems as well, where we show that CLC can help to improve performance by up to 6.7% over baselines. We make OD3 publicly available at https://github.com/amazon-science/amazon-od3 .

{{</citation>}}


### (53/91) CTC Blank Triggered Dynamic Layer-Skipping for Efficient CTC-based Speech Recognition (Junfeng Hou et al., 2024)

{{<citation>}}

Junfeng Hou, Peiyao Wang, Jincheng Zhang, Meng Yang, Minwei Feng, Jingcheng Yin. (2024)  
**CTC Blank Triggered Dynamic Layer-Skipping for Efficient CTC-based Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.02046v1)  

---


**ABSTRACT**  
Deploying end-to-end speech recognition models with limited computing resources remains challenging, despite their impressive performance. Given the gradual increase in model size and the wide range of model applications, selectively executing model components for different inputs to improve the inference efficiency is of great interest. In this paper, we propose a dynamic layer-skipping method that leverages the CTC blank output from intermediate layers to trigger the skipping of the last few encoder layers for frames with high blank probabilities. Furthermore, we factorize the CTC output distribution and perform knowledge distillation on intermediate layers to reduce computation and improve recognition accuracy. Experimental results show that by utilizing the CTC blank, the encoder layer depth can be adjusted dynamically, resulting in 29% acceleration of the CTC model inference with minor performance degradation.

{{</citation>}}


## cs.CL (22)



### (54/91) LLaMA Pro: Progressive LLaMA with Block Expansion (Chengyue Wu et al., 2024)

{{<citation>}}

Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, Ying Shan. (2024)  
**LLaMA Pro: Progressive LLaMA with Block Expansion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.02415v1)  

---


**ABSTRACT**  
Humans generally acquire new skills without compromising the old; however, the opposite holds for Large Language Models (LLMs), e.g., from LLaMA to CodeLLaMA. To this end, we propose a new post-pretraining method for LLMs with an expansion of Transformer blocks. We tune the expanded blocks using only new corpus, efficiently and effectively improving the model's knowledge without catastrophic forgetting. In this paper, we experiment on the corpus of code and math, yielding LLaMA Pro-8.3B, a versatile foundation model initialized from LLaMA2-7B, excelling in general tasks, programming, and mathematics. LLaMA Pro and its instruction-following counterpart (LLaMA Pro-Instruct) achieve advanced performance among various benchmarks, demonstrating superiority over existing open models in the LLaMA family and the immense potential of reasoning and addressing diverse tasks as an intelligent agent. Our findings provide valuable insights into integrating natural and programming languages, laying a solid foundation for developing advanced language agents that operate effectively in various environments.

{{</citation>}}


### (55/91) TinyLlama: An Open-Source Small Language Model (Peiyuan Zhang et al., 2024)

{{<citation>}}

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu. (2024)  
**TinyLlama: An Open-Source Small Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2401.02385v1)  

---


**ABSTRACT**  
We present TinyLlama, a compact 1.1B language model pretrained on around 1 trillion tokens for approximately 3 epochs. Building on the architecture and tokenizer of Llama 2, TinyLlama leverages various advances contributed by the open-source community (e.g., FlashAttention), achieving better computational efficiency. Despite its relatively small size, TinyLlama demonstrates remarkable performance in a series of downstream tasks. It significantly outperforms existing open-source language models with comparable sizes. Our model checkpoints and code are publicly available on GitHub at https://github.com/jzhang38/TinyLlama.

{{</citation>}}


### (56/91) SPEER: Sentence-Level Planning of Long Clinical Summaries via Embedded Entity Retrieval (Griffin Adams et al., 2024)

{{<citation>}}

Griffin Adams, Jason Zucker, Noémie Elhadad. (2024)  
**SPEER: Sentence-Level Planning of Long Clinical Summaries via Embedded Entity Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2401.02369v1)  

---


**ABSTRACT**  
Clinician must write a lengthy summary each time a patient is discharged from the hospital. This task is time-consuming due to the sheer number of unique clinical concepts covered in the admission. Identifying and covering salient entities is vital for the summary to be clinically useful. We fine-tune open-source LLMs (Mistral-7B-Instruct and Zephyr-7B-\b{eta}) on the task and find that they generate incomplete and unfaithful summaries. To increase entity coverage, we train a smaller, encoder-only model to predict salient entities, which are treated as content-plans to guide the LLM. To encourage the LLM to focus on specific mentions in the source notes, we propose SPEER: Sentence-level Planning via Embedded Entity Retrieval. Specifically, we mark each salient entity span with special "{{ }}" boundary tags and instruct the LLM to retrieve marked spans before generating each sentence. Sentence-level planning acts as a form of state tracking in that the model is explicitly recording the entities it uses. We fine-tune Mistral and Zephyr variants on a large-scale, diverse dataset of ~167k in-patient hospital admissions and evaluate on 3 datasets. SPEER shows gains in both coverage and faithfulness metrics over non-guided and guided baselines.

{{</citation>}}


### (57/91) Blar-SQL: Faster, Stronger, Smaller NL2SQL (José Manuel Domínguez et al., 2024)

{{<citation>}}

José Manuel Domínguez, Benjamín Errázuriz, Patricio Daher. (2024)  
**Blar-SQL: Faster, Stronger, Smaller NL2SQL**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.02997v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have gained considerable notoriety in the field of natural language to SQL tasks (NL2SQL). In this study, we show how task decomposition can greatly benefit LLMs in database understanding and query generation in order to answer human questions with an SQL query.   We fined-tuned open source models, specifically Llama-2 and Code Llama, by combining 2 different models each designated to focus on one of two tasks in order to leverage each model's core competency to further increase the accuracy of the final SQL query.   We propose a new framework to divide the schema into chunks in order to fit more information into a limited context. Our results are comparable with those obtained by GPT-4 at the same time being 135 times smaller, 90 times faster and more than 100 times cheaper than GPT-4.

{{</citation>}}


### (58/91) Are LLMs Robust for Spoken Dialogues? (Seyed Mahed Mousavi et al., 2024)

{{<citation>}}

Seyed Mahed Mousavi, Gabriel Roccabruna, Simone Alghisi, Massimo Rizzoli, Mirco Ravanelli, Giuseppe Riccardi. (2024)  
**Are LLMs Robust for Spoken Dialogues?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2401.02297v1)  

---


**ABSTRACT**  
Large Pre-Trained Language Models have demonstrated state-of-the-art performance in different downstream tasks, including dialogue state tracking and end-to-end response generation. Nevertheless, most of the publicly available datasets and benchmarks on task-oriented dialogues focus on written conversations. Consequently, the robustness of the developed models to spoken interactions is unknown. In this work, we have evaluated the performance of LLMs for spoken task-oriented dialogues on the DSTC11 test sets. Due to the lack of proper spoken dialogue datasets, we have automatically transcribed a development set of spoken dialogues with a state-of-the-art ASR engine. We have characterized the ASR-error types and their distributions and simulated these errors in a large dataset of dialogues. We report the intrinsic (perplexity) and extrinsic (human evaluation) performance of fine-tuned GPT-2 and T5 models in two subtasks of response generation and dialogue state tracking, respectively. The results show that LLMs are not robust to spoken noise by default, however, fine-tuning/training such models on a proper dataset of spoken TODs can result in a more robust performance.

{{</citation>}}


### (59/91) Rethinking Response Evaluation from Interlocutor's Eye for Open-Domain Dialogue Systems (Yuma Tsuta et al., 2024)

{{<citation>}}

Yuma Tsuta, Naoki Yoshinaga, Shoetsu Sato, Masashi Toyoda. (2024)  
**Rethinking Response Evaluation from Interlocutor's Eye for Open-Domain Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Twitter  
[Paper Link](http://arxiv.org/abs/2401.02256v1)  

---


**ABSTRACT**  
Open-domain dialogue systems have started to engage in continuous conversations with humans. Those dialogue systems are required to be adjusted to the human interlocutor and evaluated in terms of their perspective. However, it is questionable whether the current automatic evaluation methods can approximate the interlocutor's judgments. In this study, we analyzed and examined what features are needed in an automatic response evaluator from the interlocutor's perspective. The first experiment on the Hazumi dataset revealed that interlocutor awareness plays a critical role in making automatic response evaluation correlate with the interlocutor's judgments. The second experiment using massive conversations on X (formerly Twitter) confirmed that dialogue continuity prediction can train an interlocutor-aware response evaluator without human feedback while revealing the difficulty in evaluating generated responses compared to human responses.

{{</citation>}}


### (60/91) L3Cube-IndicNews: News-based Short Text and Long Document Classification Datasets in Indic Languages (Aishwarya Mirashi et al., 2024)

{{<citation>}}

Aishwarya Mirashi, Srushti Sonavane, Purva Lingayat, Tejas Padhiyar, Raviraj Joshi. (2024)  
**L3Cube-IndicNews: News-based Short Text and Long Document Classification Datasets in Indic Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.02254v1)  

---


**ABSTRACT**  
In this work, we introduce L3Cube-IndicNews, a multilingual text classification corpus aimed at curating a high-quality dataset for Indian regional languages, with a specific focus on news headlines and articles. We have centered our work on 10 prominent Indic languages, including Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Kannada, Odia, Malayalam, and Punjabi. Each of these news datasets comprises 10 or more classes of news articles. L3Cube-IndicNews offers 3 distinct datasets tailored to handle different document lengths that are classified as: Short Headlines Classification (SHC) dataset containing the news headline and news category, Long Document Classification (LDC) dataset containing the whole news article and the news category, and Long Paragraph Classification (LPC) containing sub-articles of the news and the news category. We maintain consistent labeling across all 3 datasets for in-depth length-based analysis. We evaluate each of these Indic language datasets using 4 different models including monolingual BERT, multilingual Indic Sentence BERT (IndicSBERT), and IndicBERT. This research contributes significantly to expanding the pool of available text classification datasets and also makes it possible to develop topic classification models for Indian regional languages. This also serves as an excellent resource for cross-lingual analysis owing to the high overlap of labels among languages. The datasets and models are shared publicly at https://github.com/l3cube-pune/indic-nlp

{{</citation>}}


### (61/91) CANAMRF: An Attention-Based Model for Multimodal Depression Detection (Yuntao Wei et al., 2024)

{{<citation>}}

Yuntao Wei, Yuzhe Zhang, Shuyang Zhang, Hong Zhang. (2024)  
**CANAMRF: An Attention-Based Model for Multimodal Depression Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL, eess-IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.02995v1)  

---


**ABSTRACT**  
Multimodal depression detection is an important research topic that aims to predict human mental states using multimodal data. Previous methods treat different modalities equally and fuse each modality by na\"ive mathematical operations without measuring the relative importance between them, which cannot obtain well-performed multimodal representations for downstream depression tasks. In order to tackle the aforementioned concern, we present a Cross-modal Attention Network with Adaptive Multi-modal Recurrent Fusion (CANAMRF) for multimodal depression detection. CANAMRF is constructed by a multimodal feature extractor, an Adaptive Multimodal Recurrent Fusion module, and a Hybrid Attention Module. Through experimentation on two benchmark datasets, CANAMRF demonstrates state-of-the-art performance, underscoring the effectiveness of our proposed approach.

{{</citation>}}


### (62/91) Joint Multi-Facts Reasoning Network For Complex Temporal Question Answering Over Knowledge Graph (Rikui Huang et al., 2024)

{{<citation>}}

Rikui Huang, Wei Wei, Xiaoye Qu, Wenfeng Xie, Xianling Mao, Dangyang Chen. (2024)  
**Joint Multi-Facts Reasoning Network For Complex Temporal Question Answering Over Knowledge Graph**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.02212v1)  

---


**ABSTRACT**  
Temporal Knowledge Graph (TKG) is an extension of regular knowledge graph by attaching the time scope. Existing temporal knowledge graph question answering (TKGQA) models solely approach simple questions, owing to the prior assumption that each question only contains a single temporal fact with explicit/implicit temporal constraints. Hence, they perform poorly on questions which own multiple temporal facts. In this paper, we propose \textbf{\underline{J}}oint \textbf{\underline{M}}ulti \textbf{\underline{F}}acts \textbf{\underline{R}}easoning \textbf{\underline{N}}etwork (JMFRN), to jointly reasoning multiple temporal facts for accurately answering \emph{complex} temporal questions. Specifically, JMFRN first retrieves question-related temporal facts from TKG for each entity of the given complex question. For joint reasoning, we design two different attention (\ie entity-aware and time-aware) modules, which are suitable for universal settings, to aggregate entities and timestamps information of retrieved facts. Moreover, to filter incorrect type answers, we introduce an additional answer type discrimination task. Extensive experiments demonstrate our proposed method significantly outperforms the state-of-art on the well-known complex temporal question benchmark TimeQuestions.

{{</citation>}}


### (63/91) DIALIGHT: Lightweight Multilingual Development and Evaluation of Task-Oriented Dialogue Systems with Large Language Models (Songbo Hu et al., 2024)

{{<citation>}}

Songbo Hu, Xiaobin Wang, Zhangdie Yuan, Anna Korhonen, Ivan Vulić. (2024)  
**DIALIGHT: Lightweight Multilingual Development and Evaluation of Task-Oriented Dialogue Systems with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model, Multilingual, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2401.02208v1)  

---


**ABSTRACT**  
We present DIALIGHT, a toolkit for developing and evaluating multilingual Task-Oriented Dialogue (ToD) systems which facilitates systematic evaluations and comparisons between ToD systems using fine-tuning of Pretrained Language Models (PLMs) and those utilising the zero-shot and in-context learning capabilities of Large Language Models (LLMs). In addition to automatic evaluation, this toolkit features (i) a secure, user-friendly web interface for fine-grained human evaluation at both local utterance level and global dialogue level, and (ii) a microservice-based backend, improving efficiency and scalability. Our evaluations reveal that while PLM fine-tuning leads to higher accuracy and coherence, LLM-based systems excel in producing diverse and likeable responses. However, we also identify significant challenges of LLMs in adherence to task-specific instructions and generating outputs in multiple languages, highlighting areas for future research. We hope this open-sourced toolkit will serve as a valuable resource for researchers aiming to develop and properly evaluate multilingual ToD systems and will lower, currently still high, entry barriers in the field.

{{</citation>}}


### (64/91) Location Aware Modular Biencoder for Tourism Question Answering (Haonan Li et al., 2024)

{{<citation>}}

Haonan Li, Martin Tomko, Timothy Baldwin. (2024)  
**Location Aware Modular Biencoder for Tourism Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.02187v1)  

---


**ABSTRACT**  
Answering real-world tourism questions that seek Point-of-Interest (POI) recommendations is challenging, as it requires both spatial and non-spatial reasoning, over a large candidate pool. The traditional method of encoding each pair of question and POI becomes inefficient when the number of candidates increases, making it infeasible for real-world applications. To overcome this, we propose treating the QA task as a dense vector retrieval problem, where we encode questions and POIs separately and retrieve the most relevant POIs for a question by utilizing embedding space similarity. We use pretrained language models (PLMs) to encode textual information, and train a location encoder to capture spatial information of POIs. Experiments on a real-world tourism QA dataset demonstrate that our approach is effective, efficient, and outperforms previous methods across all metrics. Enabled by the dense retrieval architecture, we further build a global evaluation baseline, expanding the search space by 20 times compared to previous work. We also explore several factors that impact on the model's performance through follow-up experiments. Our code and model are publicly available at https://github.com/haonan-li/LAMB.

{{</citation>}}


### (65/91) Shayona@SMM4H23: COVID-19 Self diagnosis classification using BERT and LightGBM models (Rushi Chavda et al., 2024)

{{<citation>}}

Rushi Chavda, Darshan Makwana, Vraj Patel, Anupam Shukla. (2024)  
**Shayona@SMM4H23: COVID-19 Self diagnosis classification using BERT and LightGBM models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.02158v1)  

---


**ABSTRACT**  
This paper describes approaches and results for shared Task 1 and 4 of SMMH4-23 by Team Shayona. Shared Task-1 was binary classification of english tweets self-reporting a COVID-19 diagnosis, and Shared Task-4 was Binary classification of English Reddit posts self-reporting a social anxiety disorder diagnosis. Our team has achieved the highest f1-score 0.94 in Task-1 among all participants. We have leveraged the Transformer model (BERT) in combination with the LightGBM model for both tasks.

{{</citation>}}


### (66/91) Exploring Boundary of GPT-4V on Marine Analysis: A Preliminary Case Study (Ziqiang Zheng et al., 2024)

{{<citation>}}

Ziqiang Zheng, Yiwei Chen, Jipeng Zhang, Tuan-Anh Vu, Huimin Zeng, Yue Him Wong Tim, Sai-Kit Yeung. (2024)  
**Exploring Boundary of GPT-4V on Marine Analysis: A Preliminary Case Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.02147v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated a powerful ability to answer various queries as a general-purpose assistant. The continuous multi-modal large language models (MLLM) empower LLMs with the ability to perceive visual signals. The launch of GPT-4 (Generative Pre-trained Transformers) has generated significant interest in the research communities. GPT-4V(ison) has demonstrated significant power in both academia and industry fields, as a focal point in a new artificial intelligence generation. Though significant success was achieved by GPT-4V, exploring MLLMs in domain-specific analysis (e.g., marine analysis) that required domain-specific knowledge and expertise has gained less attention. In this study, we carry out the preliminary and comprehensive case study of utilizing GPT-4V for marine analysis. This report conducts a systematic evaluation of existing GPT-4V, assessing the performance of GPT-4V on marine research and also setting a new standard for future developments in MLLMs. The experimental results of GPT-4V show that the responses generated by GPT-4V are still far away from satisfying the domain-specific requirements of the marine professions. All images and prompts used in this study will be available at https://github.com/hkust-vgd/Marine_GPT-4V_Eval

{{</citation>}}


### (67/91) DCR-Consistency: Divide-Conquer-Reasoning for Consistency Evaluation and Improvement of Large Language Models (Wendi Cui et al., 2024)

{{<citation>}}

Wendi Cui, Jiaxin Zhang, Zhuohang Li, Lopez Damien, Kamalika Das, Bradley Malin, Sricharan Kumar. (2024)  
**DCR-Consistency: Divide-Conquer-Reasoning for Consistency Evaluation and Improvement of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, BERT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.02132v1)  

---


**ABSTRACT**  
Evaluating the quality and variability of text generated by Large Language Models (LLMs) poses a significant, yet unresolved research challenge. Traditional evaluation methods, such as ROUGE and BERTScore, which measure token similarity, often fail to capture the holistic semantic equivalence. This results in a low correlation with human judgments and intuition, which is especially problematic in high-stakes applications like healthcare and finance where reliability, safety, and robust decision-making are highly critical. This work proposes DCR, an automated framework for evaluating and improving the consistency of LLM-generated texts using a divide-conquer-reasoning approach. Unlike existing LLM-based evaluators that operate at the paragraph level, our method employs a divide-and-conquer evaluator (DCE) that breaks down the paragraph-to-paragraph comparison between two generated responses into individual sentence-to-paragraph comparisons, each evaluated based on predefined criteria. To facilitate this approach, we introduce an automatic metric converter (AMC) that translates the output from DCE into an interpretable numeric score. Beyond the consistency evaluation, we further present a reason-assisted improver (RAI) that leverages the analytical reasons with explanations identified by DCE to generate new responses aimed at reducing these inconsistencies. Through comprehensive and systematic empirical analysis, we show that our approach outperforms state-of-the-art methods by a large margin (e.g., +19.3% and +24.3% on the SummEval dataset) in evaluating the consistency of LLM generation across multiple benchmarks in semantic, factual, and summarization consistency tasks. Our approach also substantially reduces nearly 90% of output inconsistencies, showing promise for effective hallucination mitigation.

{{</citation>}}


### (68/91) PEFT for Speech: Unveiling Optimal Placement, Merging Strategies, and Ensemble Techniques (Tzu-Han Lin et al., 2024)

{{<citation>}}

Tzu-Han Lin, How-Shing Wang, Hao-Yung Weng, Kuang-Chen Peng, Zih-Ching Chen, Hung-yi Lee. (2024)  
**PEFT for Speech: Unveiling Optimal Placement, Merging Strategies, and Ensemble Techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.02122v1)  

---


**ABSTRACT**  
Parameter-Efficient Fine-Tuning (PEFT) is increasingly recognized as an effective method in speech processing. However, the optimal approach and the placement of PEFT methods remain inconclusive. Our study conducts extensive experiments to compare different PEFT methods and their layer-wise placement adapting Differentiable Architecture Search (DARTS). We also explore the use of ensemble learning to leverage diverse PEFT strategies. The results reveal that DARTS does not outperform the baseline approach, which involves inserting the same PEFT method into all layers of a Self-Supervised Learning (SSL) model. In contrast, an ensemble learning approach, particularly one employing majority voting, demonstrates superior performance. Our statistical evidence indicates that different PEFT methods learn in varied ways. This variation might explain why the synergistic integration of various PEFT methods through ensemble learning can harness their unique learning capabilities more effectively compared to individual layer-wise optimization.

{{</citation>}}


### (69/91) Blending Is All You Need: Cheaper, Better Alternative to Trillion-Parameters LLM (Xiaoding Lu et al., 2024)

{{<citation>}}

Xiaoding Lu, Adian Liusie, Vyas Raina, Yuwen Zhang, William Beauchamp. (2024)  
**Blending Is All You Need: Cheaper, Better Alternative to Trillion-Parameters LLM**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.02994v2)  

---


**ABSTRACT**  
In conversational AI research, there's a noticeable trend towards developing models with a larger number of parameters, exemplified by models like ChatGPT. While these expansive models tend to generate increasingly better chat responses, they demand significant computational resources and memory. This study explores a pertinent question: Can a combination of smaller models collaboratively achieve comparable or enhanced performance relative to a singular large model? We introduce an approach termed "blending", a straightforward yet effective method of integrating multiple chat AIs. Our empirical evidence suggests that when specific smaller models are synergistically blended, they can potentially outperform or match the capabilities of much larger counterparts. For instance, integrating just three models of moderate size (6B/13B paramaeters) can rival or even surpass the performance metrics of a substantially larger model like ChatGPT (175B+ paramaters). This hypothesis is rigorously tested using A/B testing methodologies with a large user base on the Chai research platform over a span of thirty days. The findings underscore the potential of the "blending" strategy as a viable approach for enhancing chat AI efficacy without a corresponding surge in computational demands.

{{</citation>}}


### (70/91) Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion (Shangyu Wu et al., 2024)

{{<citation>}}

Shangyu Wu, Ying Xiong, Yufei Cui, Xue Liu, Buzhou Tang, Tei-Wei Kuo, Chun Jason Xue. (2024)  
**Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2401.02993v1)  

---


**ABSTRACT**  
Retrieval-based augmentations that aim to incorporate knowledge from an external database into language models have achieved great success in various knowledge-intensive (KI) tasks, such as question-answering and text generation. However, integrating retrievals in non-knowledge-intensive (NKI) tasks, such as text classification, is still challenging. Existing works focus on concatenating retrievals to inputs as context to form the prompt-based inputs. Unfortunately, such methods require language models to have the capability to handle long texts. Besides, inferring such concatenated data would also consume a significant amount of computational resources.   To solve these challenges, we propose \textbf{ReFusion} in this paper, a computation-efficient \textbf{Re}trieval representation \textbf{Fusion} with neural architecture search. The main idea is to directly fuse the retrieval representations into the language models. Specifically, we first propose an online retrieval module that retrieves representations of similar sentences. Then, we present a retrieval fusion module including two effective ranking schemes, i.e., reranker-based scheme and ordered-mask-based scheme, to fuse the retrieval representations with hidden states. Furthermore, we use Neural Architecture Search (NAS) to seek the optimal fusion structure across different layers. Finally, we conduct comprehensive experiments, and the results demonstrate our ReFusion can achieve superior and robust performance on various NKI tasks.

{{</citation>}}


### (71/91) Advanced Unstructured Data Processing for ESG Reports: A Methodology for Structured Transformation and Enhanced Analysis (Jiahui Peng et al., 2024)

{{<citation>}}

Jiahui Peng, Jing Gao, Xin Tong, Jing Guo, Hang Yang, Jianchuan Qi, Ruiqiao Li, Nan Li, Ming Xu. (2024)  
**Advanced Unstructured Data Processing for ESG Reports: A Methodology for Structured Transformation and Enhanced Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2401.02992v1)  

---


**ABSTRACT**  
In the evolving field of corporate sustainability, analyzing unstructured Environmental, Social, and Governance (ESG) reports is a complex challenge due to their varied formats and intricate content. This study introduces an innovative methodology utilizing the "Unstructured Core Library", specifically tailored to address these challenges by transforming ESG reports into structured, analyzable formats. Our approach significantly advances the existing research by offering high-precision text cleaning, adept identification and extraction of text from images, and standardization of tables within these reports. Emphasizing its capability to handle diverse data types, including text, images, and tables, the method adeptly manages the nuances of differing page layouts and report styles across industries. This research marks a substantial contribution to the fields of industrial ecology and corporate sustainability assessment, paving the way for the application of advanced NLP technologies and large language models in the analysis of corporate governance and sustainability. Our code is available at https://github.com/linancn/TianGong-AI-Unstructure.git.

{{</citation>}}


### (72/91) ICE-GRT: Instruction Context Enhancement by Generative Reinforcement based Transformers (Chen Zheng et al., 2024)

{{<citation>}}

Chen Zheng, Ke Sun, Da Tang, Yukun Ma, Yuyu Zhang, Chenguang Xi, Xun Zhou. (2024)  
**ICE-GRT: Instruction Context Enhancement by Generative Reinforcement based Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model, Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.02072v1)  

---


**ABSTRACT**  
The emergence of Large Language Models (LLMs) such as ChatGPT and LLaMA encounter limitations in domain-specific tasks, with these models often lacking depth and accuracy in specialized areas, and exhibiting a decrease in general capabilities when fine-tuned, particularly analysis ability in small sized models. To address these gaps, we introduce ICE-GRT, utilizing Reinforcement Learning from Human Feedback (RLHF) grounded in Proximal Policy Optimization (PPO), demonstrating remarkable ability in in-domain scenarios without compromising general task performance. Our exploration of ICE-GRT highlights its understanding and reasoning ability to not only generate robust answers but also to provide detailed analyses of the reasons behind the answer. This capability marks a significant progression beyond the scope of Supervised Fine-Tuning models. The success of ICE-GRT is dependent on several crucial factors, including Appropriate Data, Reward Size Scaling, KL-Control, Advantage Normalization, etc. The ICE-GRT model exhibits state-of-the-art performance in domain-specific tasks and across 12 general Language tasks against equivalent size and even larger size LLMs, highlighting the effectiveness of our approach. We provide a comprehensive analysis of the ICE-GRT, underscoring the significant advancements it brings to the field of LLM.

{{</citation>}}


### (73/91) Understanding LLMs: A Comprehensive Overview from Training to Inference (Yiheng Liu et al., 2024)

{{<citation>}}

Yiheng Liu, Hao He, Tianle Han, Xu Zhang, Mengyuan Liu, Jiaming Tian, Yutong Zhang, Jiaqi Wang, Xiaohui Gao, Tianyang Zhong, Yi Pan, Shaochen Xu, Zihao Wu, Zhengliang Liu, Xin Zhang, Shu Zhang, Xintao Hu, Tuo Zhang, Ning Qiang, Tianming Liu, Bao Ge. (2024)  
**Understanding LLMs: A Comprehensive Overview from Training to Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.02038v2)  

---


**ABSTRACT**  
The introduction of ChatGPT has led to a significant increase in the utilization of Large Language Models (LLMs) for addressing downstream tasks. There's an increasing focus on cost-efficient training and deployment within this context. Low-cost training and deployment of LLMs represent the future development trend. This paper reviews the evolution of large language model training techniques and inference deployment technologies aligned with this emerging trend. The discussion on training includes various aspects, including data preprocessing, training architecture, pre-training tasks, parallel training, and relevant content related to model fine-tuning. On the inference side, the paper covers topics such as model compression, parallel computation, memory scheduling, and structural optimization. It also explores LLMs' utilization and provides insights into their future development.

{{</citation>}}


### (74/91) Text2MDT: Extracting Medical Decision Trees from Medical Texts (Wei Zhu et al., 2024)

{{<citation>}}

Wei Zhu, Wenfeng Li, Xing Tian, Pengfei Wang, Xiaoling Wang, Jin Chen, Yuanbin Wu, Yuan Ni, Guotong Xie. (2024)  
**Text2MDT: Extracting Medical Decision Trees from Medical Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.02034v1)  

---


**ABSTRACT**  
Knowledge of the medical decision process, which can be modeled as medical decision trees (MDTs), is critical to build clinical decision support systems. However, the current MDT construction methods rely heavily on time-consuming and laborious manual annotation. In this work, we propose a novel task, Text2MDT, to explore the automatic extraction of MDTs from medical texts such as medical guidelines and textbooks. We normalize the form of the MDT and create an annotated Text-to-MDT dataset in Chinese with the participation of medical experts. We investigate two different methods for the Text2MDT tasks: (a) an end-to-end framework which only relies on a GPT style large language models (LLM) instruction tuning to generate all the node information and tree structures. (b) The pipeline framework which decomposes the Text2MDT task to three subtasks. Experiments on our Text2MDT dataset demonstrate that: (a) the end-to-end method basd on LLMs (7B parameters or larger) show promising results, and successfully outperform the pipeline methods. (b) The chain-of-thought (COT) prompting method \cite{Wei2022ChainOT} can improve the performance of the fine-tuned LLMs on the Text2MDT test set. (c) the lightweight pipelined method based on encoder-based pretrained models can perform comparably with LLMs with model complexity two magnititudes smaller. Our Text2MDT dataset is open-sourced at \url{https://tianchi.aliyun.com/dataset/95414}, and the source codes are open-sourced at \url{https://github.com/michael-wzhu/text2dt}.

{{</citation>}}


### (75/91) Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives (Wenqi Zhang et al., 2024)

{{<citation>}}

Wenqi Zhang, Yongliang Shen, Linjuan Wu, Qiuying Peng, Jun Wang, Yueting Zhuang, Weiming Lu. (2024)  
**Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.02009v1)  

---


**ABSTRACT**  
The reflection capacity of Large Language Model (LLM) has garnered extensive attention. A post-hoc prompting strategy, e.g., reflexion and self-refine, refines LLM's response based on self-evaluated or external feedback. However, recent research indicates without external feedback, LLM's intrinsic reflection is unstable. Our investigation unveils that the key bottleneck is the quality of the self-evaluated feedback. We find LLMs often exhibit overconfidence or high randomness when self-evaluate, offering stubborn or inconsistent feedback, which causes poor reflection. To remedy this, we advocate Self-Contrast: It adaptively explores diverse solving perspectives tailored to the request, contrasts the differences, and summarizes these discrepancies into a checklist which could be used to re-examine and eliminate discrepancies. Our method endows LLM with diverse perspectives to alleviate stubborn biases. Moreover, their discrepancies indicate potential errors or inherent uncertainties that LLM often overlooks. Reflecting upon these can catalyze more accurate and stable reflection. Experiments conducted on a series of reasoning and translation tasks with different LLMs serve to underscore the effectiveness and generality of our strategy.

{{</citation>}}


## cs.CY (4)



### (76/91) Correctness Comparison of ChatGPT-4, Bard, Claude-2, and Copilot for Spatial Tasks (Hartwig H. Hochmair et al., 2024)

{{<citation>}}

Hartwig H. Hochmair, Levente Juhasz, Takoda Kemp. (2024)  
**Correctness Comparison of ChatGPT-4, Bard, Claude-2, and Copilot for Spatial Tasks**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, GPT-4, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.02404v2)  

---


**ABSTRACT**  
Generative AI including large language models (LLMs) have recently gained significant interest in the geo-science community through its versatile task-solving capabilities including coding, spatial computations, generation of sample data, time-series forecasting, toponym recognition, or image classification. So far, the assessment of LLMs for spatial tasks has primarily focused on ChatGPT, arguably the most prominent AI chatbot, whereas other chatbots received less attention. To narrow this research gap, this study evaluates the correctness of responses for a set of 54 spatial tasks assigned to four prominent chatbots, i.e., ChatGPT-4, Bard, Claude-2, and Copilot. Overall, the chatbots performed well on spatial literacy, GIS theory, and interpretation of programming code and given functions, but revealed weaknesses in mapping, code generation, and code translation. ChatGPT-4 outperformed other chatbots across most task categories.

{{</citation>}}


### (77/91) Analyzing Misinformation Claims During the 2022 Brazilian General Election on WhatsApp, Twitter, and Kwai (Scott A. Hale et al., 2024)

{{<citation>}}

Scott A. Hale, Adriano Belisario, Ahmed Mostafa, Chico Camargo. (2024)  
**Analyzing Misinformation Claims During the 2022 Brazilian General Election on WhatsApp, Twitter, and Kwai**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.02395v1)  

---


**ABSTRACT**  
This study analyzes misinformation from WhatsApp, Twitter, and Kwai during the 2022 Brazilian general election. Given the democratic importance of accurate information during elections, multiple fact-checking organizations collaborated to identify and respond to misinformation via WhatsApp tiplines and power a fact-checking feature within a chatbot operated by Brazil's election authority, the TSE. WhatsApp is installed on over 99% of smartphones in Brazil, and the TSE chatbot was used by millions of citizens in the run-up to the elections. During the same period, we collected social media data from Twitter (now X) and Kwai (a popular video-sharing app similar to TikTok). Using the WhatsApp, Kwai, and Twitter data along with fact-checks from three Brazilian fact-checking organizations, we find unique claims on each platform. Even when the same claims are present on different platforms, they often differ in format, detail, length, or other characteristics. Our research highlights the limitations of current claim matching algorithms to match claims across platforms with such differences and identifies areas for further algorithmic development. Finally, we perform a descriptive analysis examining the formats (image, video, audio, text) and content themes of popular misinformation claims.

{{</citation>}}


### (78/91) Characterizing Fake News Targeting Corporations (Ke Zhou et al., 2024)

{{<citation>}}

Ke Zhou, Sanja Scepanovic, Daniele Quercia. (2024)  
**Characterizing Fake News Targeting Corporations**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2401.02191v1)  

---


**ABSTRACT**  
Misinformation proliferates in the online sphere, with evident impacts on the political and social realms, influencing democratic discourse and posing risks to public health and safety. The corporate world is also a prime target for fake news dissemination. While recent studies have attempted to characterize corporate misinformation and its effects on companies, their findings often suffer from limitations due to qualitative or narrative approaches and a narrow focus on specific industries. To address this gap, we conducted an analysis utilizing social media quantitative methods and crowd-sourcing studies to investigate corporate misinformation across a diverse array of industries within the S\&P 500 companies. Our study reveals that corporate misinformation encompasses topics such as products, politics, and societal issues. We discovered companies affected by fake news also get reputable news coverage but less social media attention, leading to heightened negativity in social media comments, diminished stock growth, and increased stress mentions among employee reviews. Additionally, we observe that a company is not targeted by fake news all the time, but there are particular times when a critical mass of fake news emerges. These findings hold significant implications for regulators, business leaders, and investors, emphasizing the necessity to vigilantly monitor the escalating phenomenon of corporate misinformation.

{{</citation>}}


### (79/91) The Compute Divide in Machine Learning: A Threat to Academic Contribution and Scrutiny? (Tamay Besiroglu et al., 2024)

{{<citation>}}

Tamay Besiroglu, Sage Andrus Bergerson, Amelia Michael, Lennart Heim, Xueyun Luo, Neil Thompson. (2024)  
**The Compute Divide in Machine Learning: A Threat to Academic Contribution and Scrutiny?**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02452v2)  

---


**ABSTRACT**  
There are pronounced differences in the extent to which industrial and academic AI labs use computing resources. We provide a data-driven survey of the role of the compute divide in shaping machine learning research. We show that a compute divide has coincided with a reduced representation of academic-only research teams in compute intensive research topics, especially foundation models. We argue that, academia will likely play a smaller role in advancing the associated techniques, providing critical evaluation and scrutiny, and in the diffusion of such models. Concurrent with this change in research focus, there is a noticeable shift in academic research towards embracing open source, pre-trained models developed within the industry. To address the challenges arising from this trend, especially reduced scrutiny of influential models, we recommend approaches aimed at thoughtfully expanding academic insights. Nationally-sponsored computing infrastructure coupled with open science initiatives could judiciously boost academic compute access, prioritizing research on interpretability, safety and security. Structured access programs and third-party auditing may also allow measured external evaluation of industry systems.

{{</citation>}}


## q-fin.TR (1)



### (80/91) Opinion formation in the world trade network (Célestin Coquidé et al., 2024)

{{<citation>}}

Célestin Coquidé, José Lages, Dima L. Shepelyansky. (2024)  
**Opinion formation in the world trade network**  

---
Primary Category: q-fin.TR  
Categories: cond-mat-stat-mech, cs-SI, physics-soc-ph, q-fin-TR, q-fin.TR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.02378v1)  

---


**ABSTRACT**  
We extend the opinion formation approach to probe the world influence of economical organizations. Our opinion formation model mimics a battle between currencies within the international trade network. Based on the United Nations Comtrade database, we construct the world trade network for the years of the last decade from 2010 to 2020. We consider different core groups constituted by countries preferring to trade in a specific currency. We will consider principally two core groups, namely, 5 Anglo-Saxon countries which prefer to trade in US dollar and the 11 BRICS+ which prefer to trade in a hypothetical currency, hereafter called BRI, pegged to their economies. We determine the trade currency preference of the other countries via a Monte Carlo process depending on the direct transactions between the countries. The results obtained in the frame of this mathematical model show that starting from year 2014 the majority of the world countries would have preferred to trade in BRI than USD. The Monte Carlo process reaches a steady state with 3 distinct groups: two groups of countries preferring, whatever is the initial distribution of the trade currency preferences, to trade, one in BRI and the other in USD, and a third group of countries swinging as a whole between USD and BRI depending on the initial distribution of the trade currency preferences. We also analyze the battle between USD, EUR and BRI, and present the reduced Google matrix description of the trade relations between the Anglo-Saxon countries and the BRICS+.

{{</citation>}}


## cs.RO (1)



### (81/91) AERIAL-CORE: AI-Powered Aerial Robots for Inspection and Maintenance of Electrical Power Infrastructures (Anibal Ollero et al., 2024)

{{<citation>}}

Anibal Ollero, Alejandro Suarez, Christos Papaioannidis, Ioannis Pitas, Juan M. Marredo, Viet Duong, Emad Ebeid, Vit Kratky, Martin Saska, Chloe Hanoune, Amr Afifi, Antonio Franchi, Charalampos Vourtsis, Dario Floreano, Goran Vasiljevic, Stjepan Bogdan, Alvaro Caballero, Fabio Ruggiero, Vincenzo Lippiello, Carlos Matilla, Giovanni Cioffi, Davide Scaramuzza, Jose R. Martinez-de-Dios, Begona C. Arrue, Carlos Martin, Krzysztof Zurad, Carlos Gaitan, Jacob Rodriguez, Antonio Munoz, Antidio Viguria. (2024)  
**AERIAL-CORE: AI-Powered Aerial Robots for Inspection and Maintenance of Electrical Power Infrastructures**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02343v1)  

---


**ABSTRACT**  
Large-scale infrastructures are prone to deterioration due to age, environmental influences, and heavy usage. Ensuring their safety through regular inspections and maintenance is crucial to prevent incidents that can significantly affect public safety and the environment. This is especially pertinent in the context of electrical power networks, which, while essential for energy provision, can also be sources of forest fires. Intelligent drones have the potential to revolutionize inspection and maintenance, eliminating the risks for human operators, increasing productivity, reducing inspection time, and improving data collection quality. However, most of the current methods and technologies in aerial robotics have been trialed primarily in indoor testbeds or outdoor settings under strictly controlled conditions, always within the line of sight of human operators. Additionally, these methods and technologies have typically been evaluated in isolation, lacking comprehensive integration. This paper introduces the first autonomous system that combines various innovative aerial robots. This system is designed for extended-range inspections beyond the visual line of sight, features aerial manipulators for maintenance tasks, and includes support mechanisms for human operators working at elevated heights. The paper further discusses the successful validation of this system on numerous electrical power lines, with aerial robots executing flights over 10 kilometers away from their ground control stations.

{{</citation>}}


## cs.DC (1)



### (82/91) Modern Computing: Vision and Challenges (Sukhpal Singh Gill et al., 2024)

{{<citation>}}

Sukhpal Singh Gill, Huaming Wu, Panos Patros, Carlo Ottaviani, Priyansh Arora, Victor Casamayor Pujol, David Haunschild, Ajith Kumar Parlikad, Oktay Cetinkaya, Hanan Lutfiyya, Vlado Stankovski, Ruidong Li, Yuemin Ding, Junaid Qadir, Ajith Abraham, Soumya K. Ghosh, Houbing Herbert Song, Rizos Sakellariou, Omer Rana, Joel J. P. C. Rodrigues, Salil S. Kanhere, Schahram Dustdar, Steve Uhlig, Kotagiri Ramamohanarao, Rajkumar Buyya. (2024)  
**Modern Computing: Vision and Challenges**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02469v1)  

---


**ABSTRACT**  
Over the past six decades, the computing systems field has experienced significant transformations, profoundly impacting society with transformational developments, such as the Internet and the commodification of computing. Underpinned by technological advancements, computer systems, far from being static, have been continuously evolving and adapting to cover multifaceted societal niches. This has led to new paradigms such as cloud, fog, edge computing, and the Internet of Things (IoT), which offer fresh economic and creative opportunities. Nevertheless, this rapid change poses complex research challenges, especially in maximizing potential and enhancing functionality. As such, to maintain an economical level of performance that meets ever-tighter requirements, one must understand the drivers of new model emergence and expansion, and how contemporary challenges differ from past ones. To that end, this article investigates and assesses the factors influencing the evolution of computing systems, covering established systems and architectures as well as newer developments, such as serverless computing, quantum computing, and on-device AI on edge devices. Trends emerge when one traces technological trajectory, which includes the rapid obsolescence of frameworks due to business and technical constraints, a move towards specialized systems and models, and varying approaches to centralized and decentralized control. This comprehensive review of modern computing systems looks ahead to the future of research in the field, highlighting key challenges and emerging trends, and underscoring their importance in cost-effectively driving technological progress.

{{</citation>}}


## cs.SE (1)



### (83/91) On Augmenting Scenario-Based Modeling with Generative AI (David Harel et al., 2024)

{{<citation>}}

David Harel, Guy Katz, Assaf Marron, Smadar Szekely. (2024)  
**On Augmenting Scenario-Based Modeling with Generative AI**  

---
Primary Category: cs.SE  
Categories: 68N19, cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.02245v1)  

---


**ABSTRACT**  
The manual modeling of complex systems is a daunting task; and although a plethora of methods exist that mitigate this issue, the problem remains very difficult. Recent advances in generative AI have allowed the creation of general-purpose chatbots, capable of assisting software engineers in various modeling tasks. However, these chatbots are often inaccurate, and an unstructured use thereof could result in erroneous system models. In this paper, we outline a method for the safer and more structured use of chatbots as part of the modeling process. To streamline this integration, we propose leveraging scenario-based modeling techniques, which are known to facilitate the automated analysis of models. We argue that through iterative invocations of the chatbot and the manual and automatic inspection of the resulting models, a more accurate system model can eventually be obtained. We describe favorable preliminary results, which highlight the potential of this approach.

{{</citation>}}


## math.NA (1)



### (84/91) Projections, Embeddings and Stability (Pelle Olsson, 2024)

{{<citation>}}

Pelle Olsson. (2024)  
**Projections, Embeddings and Stability**  

---
Primary Category: math.NA  
Categories: 65M06, 65M12, cs-NA, math-NA, math.NA  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.02197v1)  

---


**ABSTRACT**  
In the present work, we demonstrate how the pseudoinverse concept from linear algebra can be used to represent and analyze the boundary conditions of linear systems of partial differential equations. This approach has theoretical and practical implications; the theory applies even if the boundary operator is rank deficient, or near rank deficient. If desired, the pseudoinverse can be implemented directly using standard tools like Matlab. We also introduce a new and simplified version of the semidiscrete approximation of the linear PDE system, which completely avoids taking the time derivative of the boundary data. The stability results are valid for general, nondiagonal summation-by-parts norms. Another key result is the extension of summation-by-parts operators to multi-domains by means of carefully crafted embedding operators. No extra numerical boundary conditions are required at the grid interfaces. The aforementioned pseudoinverse allows for a compact representation of these multi-block operators, which preserves all relevant properties of the single-block operators. The embedding operators can be constructed for multiple space dimensions. Numerical results for the two-dimensional Maxwell's equations are presented, and they show very good agreement with theory.

{{</citation>}}


## cs.NE (3)



### (85/91) Human-in-the-Loop Policy Optimization for Preference-Based Multi-Objective Reinforcement Learning (Ke Li et al., 2024)

{{<citation>}}

Ke Li, Han Guo. (2024)  
**Human-in-the-Loop Policy Optimization for Preference-Based Multi-Objective Reinforcement Learning**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.02160v1)  

---


**ABSTRACT**  
Multi-objective reinforcement learning (MORL) aims to find a set of high-performing and diverse policies that address trade-offs between multiple conflicting objectives. However, in practice, decision makers (DMs) often deploy only one or a limited number of trade-off policies. Providing too many diversified trade-off policies to the DM not only significantly increases their workload but also introduces noise in multi-criterion decision-making. With this in mind, we propose a human-in-the-loop policy optimization framework for preference-based MORL that interactively identifies policies of interest. Our method proactively learns the DM's implicit preference information without requiring any a priori knowledge, which is often unavailable in real-world black-box decision scenarios. The learned preference information is used to progressively guide policy optimization towards policies of interest. We evaluate our approach against three conventional MORL algorithms that do not consider preference information and four state-of-the-art preference-based MORL algorithms on two MORL environments for robot control and smart grid management. Experimental results fully demonstrate the effectiveness of our proposed method in comparison to the other peer algorithms.

{{</citation>}}


### (86/91) An Example of Evolutionary Computation + Large Language Model Beating Human: Design of Efficient Guided Local Search (Fei Liu et al., 2024)

{{<citation>}}

Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang. (2024)  
**An Example of Evolutionary Computation + Large Language Model Beating Human: Design of Efficient Guided Local Search**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.02051v1)  

---


**ABSTRACT**  
It is often very tedious for human experts to design efficient algorithms. Recently, we have proposed a novel Algorithm Evolution using Large Language Model (AEL) framework for automatic algorithm design. AEL combines the power of a large language model and the paradigm of evolutionary computation to design, combine, and modify algorithms automatically. In this paper, we use AEL to design the guide algorithm for guided local search (GLS) to solve the well-known traveling salesman problem (TSP). AEL automatically evolves elite GLS algorithms in two days, with minimal human effort and no model training. Experimental results on 1,000 TSP20-TSP100 instances and TSPLib instances show that AEL-designed GLS outperforms state-of-the-art human-designed GLS with the same iteration budget. It achieves a 0% gap on TSP20 and TSP50 and a 0.032% gap on TSP100 in 1,000 iterations. Our findings mark the emergence of a new era in automatic algorithm design.

{{</citation>}}


### (87/91) Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket (Zhaokun Zhou et al., 2024)

{{<citation>}}

Zhaokun Zhou, Kaiwei Che, Wei Fang, Keyu Tian, Yuesheng Zhu, Shuicheng Yan, Yonghong Tian, Li Yuan. (2024)  
**Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket**  

---
Primary Category: cs.NE  
Categories: cs-CV, cs-LG, cs-NE, cs.NE  
Keywords: Attention, ImageNet, Self-Attention, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2401.02020v1)  

---


**ABSTRACT**  
Spiking Neural Networks (SNNs), known for their biologically plausible architecture, face the challenge of limited performance. The self-attention mechanism, which is the cornerstone of the high-performance Transformer and also a biologically inspired structure, is absent in existing SNNs. To this end, we explore the potential of leveraging both self-attention capability and biological properties of SNNs, and propose a novel Spiking Self-Attention (SSA) and Spiking Transformer (Spikformer). The SSA mechanism eliminates the need for softmax and captures the sparse visual feature employing spike-based Query, Key, and Value. This sparse computation without multiplication makes SSA efficient and energy-saving. Further, we develop a Spiking Convolutional Stem (SCS) with supplementary convolutional layers to enhance the architecture of Spikformer. The Spikformer enhanced with the SCS is referred to as Spikformer V2. To train larger and deeper Spikformer V2, we introduce a pioneering exploration of Self-Supervised Learning (SSL) within the SNN. Specifically, we pre-train Spikformer V2 with masking and reconstruction style inspired by the mainstream self-supervised Transformer, and then finetune the Spikformer V2 on the image classification on ImageNet. Extensive experiments show that Spikformer V2 outperforms other previous surrogate training and ANN2SNN methods. An 8-layer Spikformer V2 achieves an accuracy of 80.38% using 4 time steps, and after SSL, a 172M 16-layer Spikformer V2 reaches an accuracy of 81.10% with just 1 time step. To the best of our knowledge, this is the first time that the SNN achieves 80+% accuracy on ImageNet. The code will be available at Spikformer V2.

{{</citation>}}


## cs.IR (1)



### (88/91) Spectral-based Graph Neutral Networks for Complementary Item Recommendation (Haitong Luo et al., 2024)

{{<citation>}}

Haitong Luo, Xuying Meng, Suhang Wang, Hanyun Cao, Weiyao Zhang, Yequan Wang, Yujun Zhang. (2024)  
**Spectral-based Graph Neutral Networks for Complementary Item Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-SI, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.02130v1)  

---


**ABSTRACT**  
Modeling complementary relationships greatly helps recommender systems to accurately and promptly recommend the subsequent items when one item is purchased. Unlike traditional similar relationships, items with complementary relationships may be purchased successively (such as iPhone and Airpods Pro), and they not only share relevance but also exhibit dissimilarity. Since the two attributes are opposites, modeling complementary relationships is challenging. Previous attempts to exploit these relationships have either ignored or oversimplified the dissimilarity attribute, resulting in ineffective modeling and an inability to balance the two attributes. Since Graph Neural Networks (GNNs) can capture the relevance and dissimilarity between nodes in the spectral domain, we can leverage spectral-based GNNs to effectively understand and model complementary relationships. In this study, we present a novel approach called Spectral-based Complementary Graph Neural Networks (SComGNN) that utilizes the spectral properties of complementary item graphs. We make the first observation that complementary relationships consist of low-frequency and mid-frequency components, corresponding to the relevance and dissimilarity attributes, respectively. Based on this spectral observation, we design spectral graph convolutional networks with low-pass and mid-pass filters to capture the low-frequency and mid-frequency components. Additionally, we propose a two-stage attention mechanism to adaptively integrate and balance the two attributes. Experimental results on four e-commerce datasets demonstrate the effectiveness of our model, with SComGNN significantly outperforming existing baseline models.

{{</citation>}}


## q-bio.BM (1)



### (89/91) ACP-ESM: A novel framework for classification of anticancer peptides using protein-oriented transformer approach (Zeynep Hilal Kilimci et al., 2024)

{{<citation>}}

Zeynep Hilal Kilimci, Mustafa Yalcin. (2024)  
**ACP-ESM: A novel framework for classification of anticancer peptides using protein-oriented transformer approach**  

---
Primary Category: q-bio.BM  
Categories: cs-AI, cs-CE, cs-LG, q-bio-BM, q-bio.BM  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.02124v1)  

---


**ABSTRACT**  
Anticancer peptides (ACPs) are a class of molecules that have gained significant attention in the field of cancer research and therapy. ACPs are short chains of amino acids, the building blocks of proteins, and they possess the ability to selectively target and kill cancer cells. One of the key advantages of ACPs is their ability to selectively target cancer cells while sparing healthy cells to a greater extent. This selectivity is often attributed to differences in the surface properties of cancer cells compared to normal cells. That is why ACPs are being investigated as potential candidates for cancer therapy. ACPs may be used alone or in combination with other treatment modalities like chemotherapy and radiation therapy. While ACPs hold promise as a novel approach to cancer treatment, there are challenges to overcome, including optimizing their stability, improving selectivity, and enhancing their delivery to cancer cells, continuous increasing in number of peptide sequences, developing a reliable and precise prediction model. In this work, we propose an efficient transformer-based framework to identify anticancer peptides for by performing accurate a reliable and precise prediction model. For this purpose, four different transformer models, namely ESM, ProtBert, BioBERT, and SciBERT are employed to detect anticancer peptides from amino acid sequences. To demonstrate the contribution of the proposed framework, extensive experiments are carried on widely-used datasets in the literature, two versions of AntiCp2, cACP-DeepGram, ACP-740. Experiment results show the usage of proposed model enhances classification accuracy when compared to the state-of-the-art studies. The proposed framework, ESM, exhibits 96.45 of accuracy for AntiCp2 dataset, 97.66 of accuracy for cACP-DeepGram dataset, and 88.51 of accuracy for ACP-740 dataset, thence determining new state-of-the-art.

{{</citation>}}


## cs.DB (1)



### (90/91) Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-Dimensional Vector Similarity Search on Data Segment (Mengzhao Wang et al., 2024)

{{<citation>}}

Mengzhao Wang, Weizhi Xu, Xiaomeng Yi, Songlin Wu, Zhangyang Peng, Xiangyu Ke, Yunjun Gao, Xiaoliang Xu, Rentong Guo, Charles Xie. (2024)  
**Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-Dimensional Vector Similarity Search on Data Segment**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs-IR, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02116v1)  

---


**ABSTRACT**  
High-dimensional vector similarity search (HVSS) is receiving a spotlight as a powerful tool for various data science and AI applications. As vector data grows larger, in-memory indexes become extremely expensive because they necessitate substantial expansion of main memory resources. One possible solution is to use disk-based implementation, which stores and searches vector data in high-performance devices like NVMe SSDs. However, HVSS for data segments is still challenging in vector databases, where one machine has multiple segments for system features (like scaling) purposes. In this setting, each segment has limited memory and disk space, so HVSS on the data segment needs to balance accuracy, efficiency, and space cost. Existing disk-based methods are sub-optimal because they do not consider all these requirements together. In this paper, we present Starling, an I/O-efficient disk-resident graph index framework that optimizes data layout and search strategy in the segment. It has two main components: (1) a data layout that includes an in-memory navigation graph and a reordered disk-based graph with locality enhancement, which reduces the search path length and disk bandwidth wastage; and (2) a block search strategy that minimizes expensive disk I/Os when executing a vector query. We conduct extensive experiments to verify Starling's effectiveness, efficiency, and scalability. On a data segment with 2GB memory and 10GB disk capacity, Starling can maintain up to 33 million vectors in 128 dimensions, and serve HVSS with more than 0.9 average precision and top-10 recall rate, and latency of under 1 millisecond. The results show that Starling exhibits 43.9$\times$ higher throughput with 98% lower query latency than state-of-the-art methods under the same accuracy.

{{</citation>}}


## stat.ML (1)



### (91/91) U-Trustworthy Models.Reliability, Competence, and Confidence in Decision-Making (Ritwik Vashistha et al., 2024)

{{<citation>}}

Ritwik Vashistha, Arya Farahi. (2024)  
**U-Trustworthy Models.Reliability, Competence, and Confidence in Decision-Making**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.02062v1)  

---


**ABSTRACT**  
With growing concerns regarding bias and discrimination in predictive models, the AI community has increasingly focused on assessing AI system trustworthiness. Conventionally, trustworthy AI literature relies on the probabilistic framework and calibration as prerequisites for trustworthiness. In this work, we depart from this viewpoint by proposing a novel trust framework inspired by the philosophy literature on trust. We present a precise mathematical definition of trustworthiness, termed $\mathcal{U}$-trustworthiness, specifically tailored for a subset of tasks aimed at maximizing a utility function. We argue that a model's $\mathcal{U}$-trustworthiness is contingent upon its ability to maximize Bayes utility within this task subset. Our first set of results challenges the probabilistic framework by demonstrating its potential to favor less trustworthy models and introduce the risk of misleading trustworthiness assessments. Within the context of $\mathcal{U}$-trustworthiness, we prove that properly-ranked models are inherently $\mathcal{U}$-trustworthy. Furthermore, we advocate for the adoption of the AUC metric as the preferred measure of trustworthiness. By offering both theoretical guarantees and experimental validation, AUC enables robust evaluation of trustworthiness, thereby enhancing model selection and hyperparameter tuning to yield more trustworthy outcomes.

{{</citation>}}
